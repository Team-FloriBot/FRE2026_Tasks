#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "visualization_msgs/msg/marker.hpp"

/**
 * Pure Pursuit Controller für Outdoor-Feldroboter mit Knicklenkung
 * 
 * Dieser Knoten implementiert einen minimalistischen Pure-Pursuit-Controller,
 * der einen über /path veröffentlichten nav_msgs/Path verfolgt und daraus
 * geometry_msgs/Twist-Nachrichten auf /cmd_vel erzeugt.
 * 
 * Features:
 * - Adaptive Lookahead-Distanz basierend auf Geschwindigkeit
 * - Krümmungsabhängige Geschwindigkeitsregelung
 * - Zielerkennung mit Stop-funktion
 * - RViz-Debug-Marker für nearest und lookahead point
 */
class PurePursuitNode : public rclcpp::Node
{
public:
  PurePursuitNode() : Node("pure_pursuit_node")
  {
    // ========== Parameter laden ==========
    declare_parameters();
    load_parameters();

    // ========== TF2-Listener vorbereiten ==========
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
    
    // ========== Subscriber erstellen ==========
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      path_topic_, 10,
      [this](const nav_msgs::msg::Path::SharedPtr msg) { path_callback(msg); });
      
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      [this](const nav_msgs::msg::Odometry::SharedPtr msg) { odom_callback(msg); });
    
    // ========== Publisher erstellen ==========
    cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("~/debug_marker", 10);
  
    
    // ========== Timer für Regelkreis ==========
    control_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / control_rate_)),
      [this]() { control_loop(); });
      
    RCLCPP_INFO(this->get_logger(), "Pure Pursuit Controller gestartet");
  }

private:
  void declare_parameters()
  {
    // Topics
    declare_parameter("path_topic", "/plan");
    declare_parameter("odom_topic", "/map");
    declare_parameter("cmd_vel_topic", "/cmd_vel");
    
    // Frames
    declare_parameter("map_frame", "map");
    declare_parameter("odom_frame", "odom");
    declare_parameter("base_link_frame", "base_link");
    
    // Pure Pursuit
    declare_parameter("lookahead_min", 0.5);
    declare_parameter("lookahead_gain", 1.0);
    
    // Geschwindigkeit
    declare_parameter("max_speed", 2.0);
    declare_parameter("min_speed", 0.3);
    declare_parameter("curvature_speed_gain", 2.0);
    
    // Ziel
    declare_parameter("goal_tolerance", 0.15);
    
    // Regelung
    declare_parameter("control_rate", 10.0);
  }
  
  void load_parameters()
  {
    // Topics
    path_topic_ = get_parameter("path_topic").as_string();
    odom_topic_ = get_parameter("odom_topic").as_string();
    cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
    
    // Frames
    map_frame_ = get_parameter("map_frame").as_string();
    odom_frame_ = get_parameter("odom_frame").as_string();
    base_link_frame_ = get_parameter("base_link_frame").as_string();
    
    // Pure Pursuit
    lookahead_min_ = get_parameter("lookahead_min").as_double();
    lookahead_gain_ = get_parameter("lookahead_gain").as_double();
    
    // Geschwindigkeit
    max_speed_ = get_parameter("max_speed").as_double();
    min_speed_ = get_parameter("min_speed").as_double();
    curvature_speed_gain_ = get_parameter("curvature_speed_gain").as_double();
    
    // Ziel
    goal_tolerance_ = get_parameter("goal_tolerance").as_double();
    
    // Regelung
    control_rate_ = get_parameter("control_rate").as_double();
  }
  
  void path_callback(const nav_msgs::msg::Path::SharedPtr msg)
  {
    // Pfad speichern (map-frame)
    current_path_ = *msg;
    path_received_ = true;
    
    // Wenn noch kein Startpunkt erreicht wurde, den nearest point setzen
    if (!start_point_set_) {
      set_start_point();
    }
  }
  
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    current_odom_ = *msg;
    odometry_received_ = true;
  }
  
  void set_start_point()
  {
    // Setze Startpunkt zum Beginn des Pfades (in base_link transformiert)
    if (current_path_.poses.empty()) {
      return;
    }
    
    try {
      auto transform = tf_buffer_->lookupTransform(
        base_link_frame_, map_frame_,
        tf2::TimePointZero);
        
      // Transformiere ersten Pfadpunkt
      geometry_msgs::msg::PoseStamped start_pose;
      start_pose.header.frame_id = map_frame_;
      start_pose.pose = current_path_.poses[0].pose;
      
      tf2::doTransform(start_pose, start_pose, transform);
      
      start_point_x_ = start_pose.pose.position.x;
      start_point_y_ = start_pose.pose.position.y;
      start_point_set_ = true;
      
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "Konnte Startpunkt nicht setzen: %s", ex.what());
    }
  }
  
  void control_loop()
  {
    // Prüfen ob Pfad empfangen wurde
    if (!path_received_) {
      return;
    }
    
    // Prüfen ob Odometrie empfangen wurde
    if (!odometry_received_) {
      return;
    }
    
    // Fahrzeugpose im map-frame holen
    geometry_msgs::msg::PoseStamped vehicle_pose_map;
    if (!get_vehicle_pose_map(vehicle_pose_map)) {
      return;
    }
    
    // Nearest point und lookahead point finden
    int nearest_idx = find_nearest_point(vehicle_pose_map);
    if (nearest_idx == -1) {
      return;
    }
    
    // Wenn Endpunkt erreicht ist, stoppen
    if (nearest_idx >= static_cast<int>(current_path_.poses.size()) - 1) {
      if (check_goal_reached(vehicle_pose_map)) {
        publish_zero_twist();
        return;
      }
    }
    
    // Lookahead point finden
    int lookahead_idx = find_lookahead_point(nearest_idx, vehicle_pose_map);
    if (lookahead_idx == -1) {
      return;
    }
    
    // Fahrzeugkoordinaten für lookahead point berechnen
    geometry_msgs::msg::PoseStamped lookahead_pose_vehicle;
    if (!transform_to_vehicle_frame(
        current_path_.poses[lookahead_idx], lookahead_pose_vehicle))
    {
      return;
    }
    
    // Pure Pursuit Berechnungen
    double lookahead_distance = compute_lookahead_distance();
    double lateral_error = lookahead_pose_vehicle.pose.position.y;
    
    // Krümmung berechnen: κ = 2y / Ld²
    double curvature = 2.0 * lateral_error / (lookahead_distance * lookahead_distance);
    
    // Geschwindigkeit basierend auf Krümmung anpassen
    double speed = compute_speed(curvature);
    
    // Winkelgeschwindigkeit berechnen: ω = v * κ
    double angular_velocity = speed * curvature;
    
    // Twist publizieren
    geometry_msgs::msg::Twist twist_msg;
    twist_msg.linear.x = speed;
    twist_msg.angular.z = angular_velocity;
    cmd_vel_pub_->publish(twist_msg);
    
    // Debug Marker publishen
    publish_debug_markers(nearest_idx, lookahead_idx);
  }
  
  bool get_vehicle_pose_map(geometry_msgs::msg::PoseStamped & pose)
  {
    try {
      // Odometrie in map-frame transformieren
      geometry_msgs::msg::PoseStamped pose_odom;
      pose_odom.header = current_odom_.header;
      pose_odom.pose = current_odom_.pose.pose;
      
      auto transform = tf_buffer_->lookupTransform(
        map_frame_, odom_frame_,
        tf2::TimePointZero);
        
      tf2::doTransform(pose_odom, pose, transform);
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "TF Transform fehlgeschlagen: %s", ex.what());
      return false;
    }
  }
  
  int find_nearest_point(const geometry_msgs::msg::PoseStamped & vehicle_pose)
  {
    int nearest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < current_path_.poses.size(); ++i) {
      double dx = current_path_.poses[i].pose.position.x - vehicle_pose.pose.position.x;
      double dy = current_path_.poses[i].pose.position.y - vehicle_pose.pose.position.y;
      double dist = std::sqrt(dx*dx + dy*dy);
      
      if (dist < min_dist) {
        min_dist = dist;
        nearest_idx = static_cast<int>(i);
      }
    }
    
    return nearest_idx;
  }
  
  int find_lookahead_point(int nearest_idx, const geometry_msgs::msg::PoseStamped & vehicle_pose)
  {
    double lookahead_distance = compute_lookahead_distance();
    
    // Abstandsvariable initialisieren
    double accumulated_dist = 0.0;
    
    // Ab nearest_idx entlang des Pfades suchen
    for (size_t i = nearest_idx; i < current_path_.poses.size(); ++i) {
      if (i == nearest_idx) {
        // Erster Punkt: Distanz zum nearest point berechnen
        double dx = current_path_.poses[i].pose.position.x - vehicle_pose.pose.position.x;
        double dy = current_path_.poses[i].pose.position.y - vehicle_pose.pose.position.y;
        accumulated_dist = std::sqrt(dx*dx + dy*dy);
      } else {
        // Distanz zum nächsten Punkt addieren
        double dx = current_path_.poses[i].pose.position.x - current_path_.poses[i-1].pose.position.x;
        double dy = current_path_.poses[i].pose.position.y - current_path_.poses[i-1].pose.position.y;
        accumulated_dist += std::sqrt(dx*dx + dy*dy);
      }
      
      // Wenn lookahead distance überschritten wurde, zurückgeben
      if (accumulated_dist >= lookahead_distance) {
        return static_cast<int>(i);
      }
    }
    
    // Wenn lookahead distance nicht erreicht wurde, letzten Punkt zurückgeben
    return static_cast<int>(current_path_.poses.size()) - 1;
  }
  
  double compute_lookahead_distance()
  {
    // Adaptive Lookahead Distanz: Ld = L_min + k_v * v
    double speed = std::sqrt(
      current_odom_.twist.twist.linear.x * current_odom_.twist.twist.linear.x +
      current_odom_.twist.twist.linear.y * current_odom_.twist.twist.linear.y);
    return lookahead_min_ + lookahead_gain_ * speed;
  }
  
  double compute_speed(double curvature)
  {
    // Geschwindigkeit basierend auf Krümmung reduzieren: v = v_max * e^(-k * |κ|)
    double abs_curvature = std::abs(curvature);
    double speed = max_speed_ * std::exp(-curvature_speed_gain_ * abs_curvature);
    
    // Begrenzung auf [min_speed, max_speed]
    return std::max(min_speed_, std::min(max_speed_, speed));
  }
  
  bool transform_to_vehicle_frame(
    const geometry_msgs::msg::PoseStamped & pose_map,
    geometry_msgs::msg::PoseStamped & pose_vehicle)
  {
    try {
      auto transform = tf_buffer_->lookupTransform(
        base_link_frame_, map_frame_,
        tf2::TimePointZero);
        
      tf2::doTransform(pose_map, pose_vehicle, transform);
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "TF Transform fehlgeschlagen: %s", ex.what());
      return false;
    }
  }
  
  bool check_goal_reached(const geometry_msgs::msg::PoseStamped & vehicle_pose)
  {
    // Distanz zum letzten Pfadpunkt berechnen
    const auto & last_pose = current_path_.poses.back().pose;
    double dx = last_pose.position.x - vehicle_pose.pose.position.x;
    double dy = last_pose.position.y - vehicle_pose.pose.position.y;
    double dist = std::sqrt(dx*dx + dy*dy);
    
    return dist < goal_tolerance_;
  }
  
  void publish_zero_twist()
  {
    geometry_msgs::msg::Twist twist_msg;
    twist_msg.linear.x = 0.0;
    twist_msg.angular.z = 0.0;
    cmd_vel_pub_->publish(twist_msg);
    
    // Debug Marker löschen
    visualization_msgs::msg::Marker marker;
    marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_pub_->publish(marker);
    
    // Pfad zurücksetzen
    path_received_ = false;
    start_point_set_ = false;
    
    RCLCPP_INFO(this->get_logger(), "Ziel erreicht - Fahrzeug gestoppt");
  }
  
  void publish_debug_markers(int nearest_idx, int lookahead_idx)
  {
    // Nearest point Marker
    visualization_msgs::msg::Marker nearest_marker;
    nearest_marker.header.frame_id = map_frame_;
    nearest_marker.header.stamp = this->now();
    nearest_marker.ns = "pure_pursuit";
    nearest_marker.id = 0;
    nearest_marker.type = visualization_msgs::msg::Marker::SPHERE;
    nearest_marker.action = visualization_msgs::msg::Marker::ADD;
    nearest_marker.pose = current_path_.poses[nearest_idx].pose;
    nearest_marker.scale.x = 0.2;
    nearest_marker.scale.y = 0.2;
    nearest_marker.scale.z = 0.2;
    nearest_marker.color.r = 1.0;
    nearest_marker.color.g = 0.0;
    nearest_marker.color.b = 0.0;
    nearest_marker.color.a = 1.0;
    marker_pub_->publish(nearest_marker);
    
    // Lookahead point Marker
    visualization_msgs::msg::Marker lookahead_marker;
    lookahead_marker.header.frame_id = map_frame_;
    lookahead_marker.header.stamp = this->now();
    lookahead_marker.ns = "pure_pursuit";
    lookahead_marker.id = 1;
    lookahead_marker.type = visualization_msgs::msg::Marker::SPHERE;
    lookahead_marker.action = visualization_msgs::msg::Marker::ADD;
    lookahead_marker.pose = current_path_.poses[lookahead_idx].pose;
    lookahead_marker.scale.x = 0.3;
    lookahead_marker.scale.y = 0.3;
    lookahead_marker.scale.z = 0.3;
    lookahead_marker.color.r = 0.0;
    lookahead_marker.color.g = 1.0;
    lookahead_marker.color.b = 0.0;
    lookahead_marker.color.a = 1.0;
    marker_pub_->publish(lookahead_marker);
  }

private:
  // ========== Member Variablen ==========
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  
  rclcpp::TimerBase::SharedPtr control_timer_;
  
  // ========== Parameter ==========
  std::string path_topic_;
  std::string odom_topic_;
  std::string cmd_vel_topic_;
  
  std::string map_frame_;
  std::string odom_frame_;
  std::string base_link_frame_;
  
  double lookahead_min_;
  double lookahead_gain_;
  
  double max_speed_;
  double min_speed_;
  double curvature_speed_gain_;
  
  double goal_tolerance_;
  double control_rate_;
  
  // ========== Daten ==========
  nav_msgs::msg::Path current_path_;
  nav_msgs::msg::Odometry current_odom_;
  
  bool path_received_ = false;
  bool odometry_received_ = false;
  bool start_point_set_ = false;
  double start_point_x_ = 0.0;
  double start_point_y_ = 0.0;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PurePursuitNode>());
  rclcpp::shutdown();
  return 0;
}
