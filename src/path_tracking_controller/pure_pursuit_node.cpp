// pure_pursuit_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <limits>
#include <algorithm>
#include <visualization_msgs/msg/marker_array.hpp>

class PurePursuitNode : public rclcpp::Node
{
public:
    PurePursuitNode() : Node("pure_pursuit_node")
    {
        // Parameter deklarieren
        this->declare_parameter<std::string>("global_frame", "odom");
        this->declare_parameter<std::string>("base_link_frame", "base_link");
        this->declare_parameter<std::string>("path_topic", "/plan");
        this->declare_parameter<double>("lookahead_min", 0.5);
        this->declare_parameter<double>("lookahead_max", 2.5);
        this->declare_parameter<double>("lookahead_gain", 1.0);
        this->declare_parameter<double>("max_speed", 2.0);
        this->declare_parameter<double>("min_speed", 0.3);
        this->declare_parameter<double>("curvature_gain", 1.5);
        this->declare_parameter<double>("max_angular_velocity", 2.0);
        this->declare_parameter<double>("goal_slowdown_distance", 2.0);
        this->declare_parameter<double>("goal_tolerance", 0.15);
        this->declare_parameter<int>("search_window", 200);
        this->declare_parameter<double>("control_rate", 20.0);

        // Parameter einlesen
        global_frame_ = this->get_parameter("global_frame").as_string();
        base_link_frame_ = this->get_parameter("base_link_frame").as_string();
        path_topic_ = this->get_parameter("path_topic").as_string();
        lookahead_min_ = this->get_parameter("lookahead_min").as_double();
        lookahead_max_ = this->get_parameter("lookahead_max").as_double();
        lookahead_gain_ = this->get_parameter("lookahead_gain").as_double();
        max_speed_ = this->get_parameter("max_speed").as_double();
        min_speed_ = this->get_parameter("min_speed").as_double();
        curvature_gain_ = this->get_parameter("curvature_gain").as_double();
        max_angular_velocity_ = this->get_parameter("max_angular_velocity").as_double();
        goal_slowdown_distance_ = this->get_parameter("goal_slowdown_distance").as_double();
        goal_tolerance_ = this->get_parameter("goal_tolerance").as_double();
        search_window_ = this->get_parameter("search_window").as_int();
        control_rate_ = this->get_parameter("control_rate").as_double();

        // Interfaces erstellen
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            path_topic_, 10, std::bind(&PurePursuitNode::path_callback, this, std::placeholders::_1));
        debug_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/pure_pursuit/debug", 10);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        std::chrono::duration<double> double_duration(1.0 / control_rate_);
        timer_ = this->create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(double_duration),
            std::bind(&PurePursuitNode::control_loop, this));
    }

private:
    // ---- ROS Interfaces ----
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ---- Parameters ----
    std::string global_frame_;
    std::string base_link_frame_;
    std::string path_topic_;
    double lookahead_min_, lookahead_max_, lookahead_gain_;
    double max_speed_, min_speed_;
    double curvature_gain_, max_angular_velocity_;
    double goal_slowdown_distance_;
    double goal_tolerance_;
    int search_window_;
    double control_rate_;

    // ---- Path & State ----
    nav_msgs::msg::Path current_path_;
    size_t current_path_index_ = 0;
    bool path_received_ = false;

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Empfangener Pfad ist leer!");
            return;
        }
        current_path_ = *msg;
        current_path_index_ = 0; 
        path_received_ = true;
        RCLCPP_INFO(this->get_logger(), "Pfad mit %zu Punkten empfangen (Frame: %s)", 
                    current_path_.poses.size(), current_path_.header.frame_id.c_str());
    }

    bool get_vehicle_pose(const std::string & target_frame, geometry_msgs::msg::PoseStamped & pose)
    {
        try
        {
            auto tf = tf_buffer_->lookupTransform(target_frame, base_link_frame_, tf2::TimePointZero);
            pose.header = tf.header;
            pose.pose.position.x = tf.transform.translation.x;
            pose.pose.position.y = tf.transform.translation.y;
            pose.pose.position.z = tf.transform.translation.z;
            pose.pose.orientation = tf.transform.rotation;
            return true;
        }
        catch(const tf2::TransformException & ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                "TF-Fehler bei Fahrzeug-Pose: %s", ex.what());
            return false;
        }
    }

    // ---- Vorwärts gerichtete Suche (KORRIGIERT FÜR SCHLEIFEN) ----
    size_t find_nearest_point_forward(const geometry_msgs::msg::PoseStamped & pose)
    {
        size_t best_idx = current_path_index_;
        double best_dist = std::numeric_limits<double>::max();
        
        size_t start_idx = current_path_index_;
        size_t end_idx = std::min(current_path_index_ + static_cast<size_t>(search_window_), current_path_.poses.size());
        
        // FIX: Wenn wir ganz am Anfang stehen, darf die Suche NICHT das Ende des Pfades prüfen.
        // Da Start- und Endpunkt identisch sind, würde er sonst sofort ans Ende springen.
        if (current_path_index_ < 10 && current_path_.poses.size() > 30)
        {
            end_idx = std::min(end_idx, current_path_.poses.size() - 20);
        }
        
        for(size_t i = start_idx; i < end_idx; ++i)
        {
            double dx = current_path_.poses[i].pose.position.x - pose.pose.position.x;
            double dy = current_path_.poses[i].pose.position.y - pose.pose.position.y;
            double dist = dx*dx + dy*dy;
            if(dist < best_dist)
            {
                best_dist = dist;
                best_idx = i;
            }
        }
        current_path_index_ = std::max(current_path_index_, best_idx);
        return current_path_index_;
    }

    double compute_lookahead_distance()
    {
        double ld = lookahead_min_ + lookahead_gain_ * max_speed_;
        return std::clamp(ld, lookahead_min_, lookahead_max_);
    }

    bool find_lookahead_point(size_t nearest_idx, double lookahead_distance, geometry_msgs::msg::Point & lookahead)
    {
        double accumulated = 0.0;
        for(size_t i=nearest_idx; i<current_path_.poses.size()-1; ++i)
        {
            const auto & p0 = current_path_.poses[i].pose.position;
            const auto & p1 = current_path_.poses[i+1].pose.position;
            double seg = std::hypot(p1.x-p0.x, p1.y-p0.y);
            if(accumulated + seg >= lookahead_distance)
            {
                double remaining = lookahead_distance - accumulated;
                double ratio = remaining / seg;
                lookahead.x = p0.x + ratio*(p1.x-p0.x);
                lookahead.y = p0.y + ratio*(p1.y-p0.y);
                lookahead.z = 0.0;
                return true;
            }
            accumulated += seg;
        }
        lookahead = current_path_.poses.back().pose.position;
        return true;
    }

    // ---- Kontrolle (KORRIGIERT FÜR SCHLEIFEN) ----
    bool compute_control(const geometry_msgs::msg::Point & lookahead_global, double & speed, double & omega)
    {
        geometry_msgs::msg::PoseStamped lookahead_pose;
        lookahead_pose.header.frame_id = current_path_.header.frame_id;
        lookahead_pose.header.stamp = this->now();
        lookahead_pose.pose.position = lookahead_global;

        geometry_msgs::msg::PoseStamped lookahead_base;
        try
        {
            auto tf = tf_buffer_->lookupTransform(base_link_frame_, current_path_.header.frame_id, tf2::TimePointZero);
            tf2::doTransform(lookahead_pose, lookahead_base, tf);
        }
        catch(const tf2::TransformException & ex)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                                "TF-Fehler bei Lookahead-Transformation: %s", ex.what());
            return false;
        }

        double x = lookahead_base.pose.position.x;
        double y = lookahead_base.pose.position.y;
        double ld = std::hypot(x,y);
        if(ld < 0.05) return false;

        double alpha = std::atan2(y,x);
        double curvature = 2.0 * std::sin(alpha) / ld;

        double v_curve = max_speed_ / (1.0 + curvature_gain_ * std::abs(curvature));
        
        // FIX: Bremsrampe vor dem Ziel nur aktivieren, wenn wir auch wirklich kurz vor dem Ende des Pfad-Arrays stehen!
        double v_goal = max_speed_;
        size_t remaining_points = current_path_.poses.size() - 1 - current_path_index_;
        if (remaining_points < 40)
        {
            double goal_dist = std::hypot(
                current_path_.poses.back().pose.position.x - current_path_.poses[current_path_index_].pose.position.x,
                current_path_.poses.back().pose.position.y - current_path_.poses[current_path_index_].pose.position.y);
            v_goal = max_speed_ * std::min(1.0, goal_dist / goal_slowdown_distance_);
        }
        
        speed = std::clamp(std::min(v_curve, v_goal), min_speed_, max_speed_);
        omega = std::clamp(speed * curvature, -max_angular_velocity_, max_angular_velocity_);
        return true;
    }

    bool check_goal_reached(const geometry_msgs::msg::PoseStamped & pose)
    {
        if (current_path_.poses.empty()) return false;
        
        size_t remaining_points = current_path_.poses.size() - 1 - current_path_index_;
        if (remaining_points > 15) 
        {
            return false;
        }

        const auto & goal = current_path_.poses.back().pose.position;
        double dist = std::hypot(goal.x - pose.pose.position.x, goal.y - pose.pose.position.y);
        return dist < goal_tolerance_;
    }

    void publish_zero_twist()
    {
        geometry_msgs::msg::Twist cmd{};
        cmd_vel_pub_->publish(cmd);
    }

    void publish_debug(
        const geometry_msgs::msg::PoseStamped & vehicle,
        const geometry_msgs::msg::Point & lookahead,
        size_t nearest_idx)
    {
        visualization_msgs::msg::MarkerArray arr;

        auto make_sphere = [&](int id, double x, double y,
                              float r, float g, float b)
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = current_path_.header.frame_id;
            m.header.stamp = this->now();
            m.ns = "pp_debug";
            m.id = id;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;

            m.pose.position.x = x;
            m.pose.position.y = y;
            m.pose.position.z = 0.0;

            m.scale.x = 0.3;
            m.scale.y = 0.3;
            m.scale.z = 0.3;

            m.color.a = 1.0;
            m.color.r = r;
            m.color.g = g;
            m.color.b = b;

            return m;
        };

        arr.markers.push_back(make_sphere(0, vehicle.pose.position.x, vehicle.pose.position.y, 1.0, 0.0, 0.0));
        arr.markers.push_back(make_sphere(1, lookahead.x, lookahead.y, 0.0, 1.0, 0.0));
        arr.markers.push_back(make_sphere(2, current_path_.poses[nearest_idx].pose.position.x, current_path_.poses[nearest_idx].pose.position.y, 0.0, 0.0, 1.0));

        debug_pub_->publish(arr);
    }

    void control_loop()
    {
        if(!path_received_ || current_path_.poses.empty()) return;

        geometry_msgs::msg::PoseStamped vehicle_pose;
        if(!get_vehicle_pose(current_path_.header.frame_id, vehicle_pose)) return;

        size_t nearest_idx = find_nearest_point_forward(vehicle_pose);

        if(check_goal_reached(vehicle_pose))
        {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Ziel erreicht! Stoppe Fahrzeug.");
            publish_zero_twist();
            return;
        }

        double ld = compute_lookahead_distance();
        geometry_msgs::msg::Point lookahead;
        if(!find_lookahead_point(nearest_idx, ld, lookahead)) return;

        double speed, omega;
        if(!compute_control(lookahead, speed, omega)) {
            publish_zero_twist(); 
            return;
        }

        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = speed;
        cmd.angular.z = omega;
        cmd_vel_pub_->publish(cmd);
        
        publish_debug(vehicle_pose, lookahead, nearest_idx);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PurePursuitNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}