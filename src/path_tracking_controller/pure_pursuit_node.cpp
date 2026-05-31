// pure_pursuit_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <visualization_msgs/msg/marker_array.hpp>

class PurePursuitNode : public rclcpp::Node
{
public:
    PurePursuitNode() : Node("pure_pursuit_node")
    {
        this->declare_parameter<double>("lookahead_min", 0.5);
        this->declare_parameter<double>("lookahead_max", 2.5);
        this->declare_parameter<double>("lookahead_gain", 1.0);
        this->declare_parameter<double>("max_speed", 2.0);
        this->declare_parameter<double>("min_speed", 0.3);
        this->declare_parameter<double>("curvature_gain", 1.5);
        this->declare_parameter<double>("max_angular_velocity", 2.0);
        this->declare_parameter<double>("control_rate", 20.0);

        lookahead_min_ = get_parameter("lookahead_min").as_double();
        lookahead_max_ = get_parameter("lookahead_max").as_double();
        lookahead_gain_ = get_parameter("lookahead_gain").as_double();
        max_speed_ = get_parameter("max_speed").as_double();
        min_speed_ = get_parameter("min_speed").as_double();
        curvature_gain_ = get_parameter("curvature_gain").as_double();
        max_angular_velocity_ = get_parameter("max_angular_velocity").as_double();
        control_rate_ = get_parameter("control_rate").as_double();

        cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        debug_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/pure_pursuit/debug", 10);
        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/plan", 10,
            std::bind(&PurePursuitNode::path_callback, this, std::placeholders::_1));

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        timer_ = create_wall_timer(
            std::chrono::milliseconds((int)(1000.0 / control_rate_)),
            std::bind(&PurePursuitNode::control_loop, this));
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    nav_msgs::msg::Path path_;
    bool path_received_ = false;

    size_t target_idx_ = 0;

    double lookahead_min_, lookahead_max_, lookahead_gain_;
    double max_speed_, min_speed_;
    double curvature_gain_, max_angular_velocity_;
    double control_rate_;

    // =========================
    // PATH CALLBACK
    // =========================
    void path_callback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.empty()) return;

        path_ = *msg;
        path_received_ = true;
        target_idx_ = 0;

        RCLCPP_INFO(get_logger(), "Path received: %zu poses", path_.poses.size());
    }

    // =========================
    // POSE
    // =========================
    bool get_robot_pose(geometry_msgs::msg::PoseStamped & pose)
    {
        try
        {
            auto tf = tf_buffer_->lookupTransform(
                path_.header.frame_id, "base_link", tf2::TimePointZero);

            pose.pose.position.x = tf.transform.translation.x;
            pose.pose.position.y = tf.transform.translation.y;
            pose.pose.orientation = tf.transform.rotation;
            return true;
        }
        catch (...) { return false; }
    }

    // =========================
    // LOOKAHEAD DIST
    // =========================
    double lookahead_distance()
    {
        return std::clamp(
            lookahead_min_ + lookahead_gain_ * max_speed_,
            lookahead_min_,
            lookahead_max_);
    }

    // =========================
    // ROUTE FOLLOWING (KEY FIX)
    // =========================
    size_t update_target_index(const geometry_msgs::msg::PoseStamped & robot)
    {
        size_t N = path_.poses.size();
        size_t best = target_idx_;
        double best_d = 1e9;

        size_t window = 50;

        for (size_t k = 0; k < window; k++)
        {
            size_t i = (target_idx_ + k) % N;

            auto & p = path_.poses[i].pose.position;

            double dx = p.x - robot.pose.position.x;
            double dy = p.y - robot.pose.position.y;
            double d = dx*dx + dy*dy;

            if (d < best_d)
            {
                best_d = d;
                best = i;
            }
        }

        target_idx_ = best;
        return best;
    }

    // =========================
    // LOOKAHEAD ON ROUTE
    // =========================
    geometry_msgs::msg::Point get_lookahead(size_t idx, double ld)
    {
        size_t N = path_.poses.size();
        double acc = 0.0;

        geometry_msgs::msg::Point out;

        for (size_t k = 0; k < N; k++)
        {
            size_t i = (idx + k) % N;
            size_t j = (i + 1) % N;

            auto & p0 = path_.poses[i].pose.position;
            auto & p1 = path_.poses[j].pose.position;

            double seg = std::hypot(p1.x - p0.x, p1.y - p0.y);
            if (seg < 1e-6) continue;

            if (acc + seg >= ld)
            {
                double r = (ld - acc) / seg;
                out.x = p0.x + r * (p1.x - p0.x);
                out.y = p0.y + r * (p1.y - p0.y);
                return out;
            }

            acc += seg;
        }

        return path_.poses[idx].pose.position;
    }

    void publish_debug(const geometry_msgs::msg::PoseStamped & robot,
                   const geometry_msgs::msg::Point & lookahead,
                   size_t idx)
    {
        visualization_msgs::msg::MarkerArray arr;

        auto mk = [&](int id, double x, double y,
                      float r, float g, float b)
        {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = path_.header.frame_id;
            m.header.stamp = this->now();

            m.ns = "pp_debug";
            m.id = id;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;

            m.pose.position.x = x;
            m.pose.position.y = y;
            m.pose.position.z = 0.0;

            m.scale.x = 0.25;
            m.scale.y = 0.25;
            m.scale.z = 0.25;

            m.color.a = 1.0;
            m.color.r = r;
            m.color.g = g;
            m.color.b = b;

            return m;
        };

        // Robot (rot)
        arr.markers.push_back(
            mk(0,
              robot.pose.position.x,
              robot.pose.position.y,
              1.0, 0.0, 0.0));

        // Lookahead (grün)
        arr.markers.push_back(
            mk(1,
              lookahead.x,
              lookahead.y,
              0.0, 1.0, 0.0));

        // Target Index (blau)
        auto & p = path_.poses[idx].pose.position;
        arr.markers.push_back(
            mk(2,
              p.x,
              p.y,
              0.0, 0.0, 1.0));

        debug_pub_->publish(arr);
    }

    // =========================
    // CONTROL
    // =========================
    void control_loop()
    {
        if (!path_received_) return;

        geometry_msgs::msg::PoseStamped robot;
        if (!get_robot_pose(robot)) return;

        size_t idx = update_target_index(robot);

        double ld = lookahead_distance();
        auto lookahead = get_lookahead(idx, ld);

        double dx = lookahead.x - robot.pose.position.x;
        double dy = lookahead.y - robot.pose.position.y;

        double alpha = std::atan2(dy, dx);
        double dist = std::hypot(dx, dy);

        if (dist < 0.05) return;

        double curvature = 2.0 * std::sin(alpha) / dist;

        double v = max_speed_ / (1.0 + curvature_gain_ * std::abs(curvature));
        v = std::clamp(v, min_speed_, max_speed_);

        double omega = std::clamp(v * curvature,
                                  -max_angular_velocity_,
                                  max_angular_velocity_);

        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = v;
        cmd.angular.z = omega;
        cmd_vel_pub_->publish(cmd);
        publish_debug(robot, lookahead, idx);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuitNode>());
    rclcpp::shutdown();
}