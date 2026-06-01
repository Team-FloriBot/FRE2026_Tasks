// pure_pursuit_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include <chrono>
#include <cmath>
#include <limits>
#include <algorithm>
#include <string>

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
        this->declare_parameter<double>("angular_acceleration_limit", 3.0);
        this->declare_parameter<double>("control_rate", 20.0);
        this->declare_parameter<double>("goal_tolerance", 0.15);
        this->declare_parameter<double>("slowdown_distance", 0.75);
        this->declare_parameter<double>("acceleration_limit", 0.5);
        this->declare_parameter<double>("deceleration_limit", 0.5);
        this->declare_parameter<std::string>("path_topic", "/plan");
        this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
        this->declare_parameter<std::string>("base_link_frame", "base_link");

        this->declare_parameter<double>("resample_spacing", 0.10);
        this->declare_parameter<bool>("resample_enabled", true);

        lookahead_min_ = get_parameter("lookahead_min").as_double();
        lookahead_max_ = get_parameter("lookahead_max").as_double();
        lookahead_gain_ = get_parameter("lookahead_gain").as_double();
        max_speed_ = get_parameter("max_speed").as_double();
        min_speed_ = get_parameter("min_speed").as_double();
        curvature_gain_ = get_parameter("curvature_gain").as_double();
        max_angular_velocity_ = get_parameter("max_angular_velocity").as_double();
        angular_acceleration_limit_ = get_parameter("angular_acceleration_limit").as_double();
        control_rate_ = get_parameter("control_rate").as_double();
        goal_tolerance_ = get_parameter("goal_tolerance").as_double();
        slowdown_distance_ = get_parameter("slowdown_distance").as_double();
        acceleration_limit_ = get_parameter("acceleration_limit").as_double();
        deceleration_limit_ = get_parameter("deceleration_limit").as_double();
        path_topic_ = get_parameter("path_topic").as_string();
        cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
        base_link_frame_ = get_parameter("base_link_frame").as_string();

        resample_spacing_ = get_parameter("resample_spacing").as_double();
        resample_enabled_ = get_parameter("resample_enabled").as_bool();

        cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
        debug_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/pure_pursuit/debug", 10);

        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            path_topic_, 10,
            std::bind(&PurePursuitNode::path_callback, this, std::placeholders::_1));

        set_active_srv_ = create_service<std_srvs::srv::SetBool>(
            "~/set_active",
            std::bind(
                &PurePursuitNode::set_active_callback,
                this,
                std::placeholders::_1,
                std::placeholders::_2));

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        timer_ = create_wall_timer(
            std::chrono::milliseconds((int)(1000.0 / control_rate_)),
            std::bind(&PurePursuitNode::control_loop, this));

        RCLCPP_INFO(
            get_logger(),
            "Pure Pursuit Controller bereit. Pfad: %s, cmd_vel: %s, Start/Stop-Service: %s",
            path_topic_.c_str(),
            cmd_vel_topic_.c_str(),
            (std::string(get_fully_qualified_name()) + "/set_active").c_str());
    }

private:
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr debug_pub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr set_active_srv_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    nav_msgs::msg::Path path_;
    bool path_received_ = false;
    bool tracking_enabled_ = false;
    bool path_completed_ = false;

    size_t target_idx_ = 0;

    double lookahead_min_, lookahead_max_, lookahead_gain_;
    double max_speed_, min_speed_;
    double curvature_gain_, max_angular_velocity_;
    double angular_acceleration_limit_;
    double control_rate_;
    double goal_tolerance_;
    double slowdown_distance_;
    double acceleration_limit_;
    double deceleration_limit_;
    double commanded_linear_speed_ = 0.0;
    double commanded_angular_velocity_ = 0.0;
    bool last_control_time_valid_ = false;
    rclcpp::Time last_control_time_;

    std::string path_topic_;
    std::string cmd_vel_topic_;
    std::string base_link_frame_;

    double resample_spacing_;
    bool resample_enabled_;

    // =========================
    // RESAMPLING
    // =========================
    nav_msgs::msg::Path resample_path(const nav_msgs::msg::Path & input)
    {
        nav_msgs::msg::Path out = input;
        out.poses.clear();

        if (input.poses.size() < 2)
            return input;

        for (size_t i = 0; i < input.poses.size() - 1; i++)
        {
            const auto & p0 = input.poses[i].pose.position;
            const auto & p1 = input.poses[i + 1].pose.position;

            double dx = p1.x - p0.x;
            double dy = p1.y - p0.y;

            double seg_len = std::hypot(dx, dy);
            if (seg_len < 1e-6)
                continue;

            int steps = std::max(1, (int)(seg_len / resample_spacing_));

            for (int s = 0; s < steps; s++)
            {
                double t = (double)s / steps;

                geometry_msgs::msg::PoseStamped ps;
                ps.header = input.header;

                ps.pose.position.x = p0.x + t * dx;
                ps.pose.position.y = p0.y + t * dy;
                ps.pose.position.z = 0.0;
                ps.pose.orientation.w = 1.0;

                out.poses.push_back(ps);
            }
        }

        // last point
        out.poses.push_back(input.poses.back());

        return out;
    }

    // =========================
    void path_callback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.size() < 2)
        {
            path_received_ = false;
            tracking_enabled_ = false;
            path_completed_ = true;
            RCLCPP_WARN(get_logger(), "Empfangener Pfad hat weniger als zwei Posen. Stoppe Path Tracking.");
            return;
        }

        if (resample_enabled_)
            path_ = resample_path(*msg);
        else
            path_ = *msg;

        path_received_ = true;
        tracking_enabled_ = true;
        path_completed_ = false;
        target_idx_ = 0;

        RCLCPP_INFO(get_logger(),
            "Path received: %zu -> %zu poses. Path Tracking gestartet/restarted.",
            msg->poses.size(),
            path_.poses.size());
    }

    void set_active_callback(
        const std_srvs::srv::SetBool::Request::SharedPtr request,
        std_srvs::srv::SetBool::Response::SharedPtr response)
    {
        if (request->data)
        {
            if (!path_received_)
            {
                response->success = false;
                response->message = "Kein Pfad vorhanden. Erst /trigger_coverage_planning aufrufen.";
                return;
            }

            tracking_enabled_ = true;
            path_completed_ = false;
            response->success = true;
            response->message = "Path Tracking gestartet.";
            RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
            return;
        }

        tracking_enabled_ = false;
        response->success = true;
        response->message = "Path Tracking wird sanft gestoppt.";
        RCLCPP_INFO(get_logger(), "%s", response->message.c_str());
    }

    // =========================
    bool get_robot_pose(geometry_msgs::msg::PoseStamped & pose)
    {
        try
        {
            auto tf = tf_buffer_->lookupTransform(
                path_.header.frame_id,
                base_link_frame_,
                tf2::TimePointZero);

            pose.pose.position.x = tf.transform.translation.x;
            pose.pose.position.y = tf.transform.translation.y;
            pose.pose.orientation = tf.transform.rotation;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    double lookahead_distance()
    {
        return std::clamp(
            lookahead_min_ + lookahead_gain_ * max_speed_,
            lookahead_min_,
            lookahead_max_);
    }

    // =========================
    size_t update_target_index(const geometry_msgs::msg::PoseStamped & robot)
    {
        size_t N = path_.poses.size();

        size_t best = target_idx_;
        double best_d = std::numeric_limits<double>::max();

        size_t window = std::min<size_t>(200, N);

        for (size_t k = 0; k < window; k++)
        {
            size_t i = std::min(target_idx_ + k, N - 1);

            const auto & p = path_.poses[i].pose.position;

            double dx = p.x - robot.pose.position.x;
            double dy = p.y - robot.pose.position.y;
            double d = dx * dx + dy * dy;

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
    geometry_msgs::msg::Point get_lookahead(size_t idx, double ld)
    {
        geometry_msgs::msg::Point out;

        double acc = 0.0;
        size_t N = path_.poses.size();

        for (size_t i = idx; i < N - 1; i++)
        {
            const auto & p0 = path_.poses[i].pose.position;
            const auto & p1 = path_.poses[i + 1].pose.position;

            double dx = p1.x - p0.x;
            double dy = p1.y - p0.y;

            double seg = std::hypot(dx, dy);
            if (seg < 1e-6)
                continue;

            if (acc + seg >= ld)
            {
                double r = (ld - acc) / seg;

                out.x = p0.x + r * dx;
                out.y = p0.y + r * dy;
                return out;
            }

            acc += seg;
        }

        return path_.poses.back().pose.position;
    }

    double remaining_path_distance(size_t idx, const geometry_msgs::msg::PoseStamped & robot)
    {
        if (path_.poses.empty())
            return 0.0;

        size_t N = path_.poses.size();
        idx = std::min(idx, N - 1);

        const auto & start = path_.poses[idx].pose.position;
        double remaining = std::hypot(
            start.x - robot.pose.position.x,
            start.y - robot.pose.position.y);

        for (size_t i = idx; i < N - 1; i++)
        {
            const auto & p0 = path_.poses[i].pose.position;
            const auto & p1 = path_.poses[i + 1].pose.position;
            remaining += std::hypot(p1.x - p0.x, p1.y - p0.y);
        }

        return remaining;
    }

    double ramp_value(double current, double target, double increase_limit, double decrease_limit, double dt)
    {
        double delta = target - current;
        double limit = (delta >= 0.0 ? increase_limit : decrease_limit) * dt;

        if (std::abs(delta) <= limit)
            return target;

        return current + std::copysign(limit, delta);
    }

    double control_dt()
    {
        rclcpp::Time now = get_clock()->now();
        double dt = 1.0 / std::max(control_rate_, 1.0);

        if (last_control_time_valid_)
        {
            dt = (now - last_control_time_).seconds();
            dt = std::clamp(dt, 0.001, 0.25);
        }

        last_control_time_ = now;
        last_control_time_valid_ = true;
        return dt;
    }

    void publish_ramped_command(double target_linear, double target_angular, double dt)
    {
        commanded_linear_speed_ = ramp_value(
            commanded_linear_speed_,
            target_linear,
            acceleration_limit_,
            deceleration_limit_,
            dt);

        commanded_angular_velocity_ = ramp_value(
            commanded_angular_velocity_,
            target_angular,
            angular_acceleration_limit_,
            angular_acceleration_limit_,
            dt);

        if (std::abs(commanded_linear_speed_) < 1e-3)
            commanded_linear_speed_ = 0.0;
        if (std::abs(commanded_angular_velocity_) < 1e-3)
            commanded_angular_velocity_ = 0.0;

        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = commanded_linear_speed_;
        cmd.angular.z = commanded_angular_velocity_;
        cmd_vel_pub_->publish(cmd);
    }

    void publish_ramped_stop(double dt)
    {
        publish_ramped_command(0.0, 0.0, dt);
    }

    // =========================
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

            m.scale.x = 0.25;
            m.scale.y = 0.25;
            m.scale.z = 0.25;

            m.color.a = 1.0;
            m.color.r = r;
            m.color.g = g;
            m.color.b = b;

            return m;
        };

        arr.markers.push_back(mk(0, robot.pose.position.x, robot.pose.position.y, 1, 0, 0));
        arr.markers.push_back(mk(1, lookahead.x, lookahead.y, 0, 1, 0));
        arr.markers.push_back(mk(2, path_.poses[idx].pose.position.x,
                                   path_.poses[idx].pose.position.y,
                                   0, 0, 1));

        debug_pub_->publish(arr);
    }

    // =========================
    void control_loop()
    {
        double dt = control_dt();

        if (!path_received_ || !tracking_enabled_ || path_completed_)
        {
            publish_ramped_stop(dt);
            return;
        }

        geometry_msgs::msg::PoseStamped robot;
        if (!get_robot_pose(robot))
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "Keine TF-Pose verfuegbar. Bremse mit Rampe auf 0.");
            publish_ramped_stop(dt);
            return;
        }

        size_t idx = update_target_index(robot);
        double remaining = remaining_path_distance(idx, robot);

        double ld = lookahead_distance();
        auto lookahead = get_lookahead(idx, ld);

        geometry_msgs::msg::PoseStamped lookahead_pose;
        lookahead_pose.header = path_.header;
        lookahead_pose.pose.position = lookahead;
        lookahead_pose.pose.orientation.w = 1.0;

        geometry_msgs::msg::PoseStamped lookahead_base;

        try
        {
            auto tf = tf_buffer_->lookupTransform(
                base_link_frame_,
                path_.header.frame_id,
                tf2::TimePointZero);

            tf2::doTransform(lookahead_pose, lookahead_base, tf);
        }
        catch (...)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "Lookahead konnte nicht transformiert werden. Bremse mit Rampe auf 0.");
            publish_ramped_stop(dt);
            return;
        }

        double x = lookahead_base.pose.position.x;
        double y = lookahead_base.pose.position.y;

        double dist = std::hypot(x, y);
        bool final_point_reached = idx >= path_.poses.size() - 1 &&
            (remaining <= goal_tolerance_ || x <= 0.0);

        if (final_point_reached)
        {
            path_completed_ = true;
            tracking_enabled_ = false;
            RCLCPP_INFO(get_logger(), "Pfadende erreicht. Bremse mit Geschwindigkeitsrampe auf 0.");
            publish_ramped_stop(dt);
            return;
        }

        if (dist < 0.05)
        {
            publish_ramped_stop(dt);
            return;
        }

        double alpha = std::atan2(y, x);

        double curvature = 2.0 * std::sin(alpha) / dist;

        double v = max_speed_ / (1.0 + curvature_gain_ * std::abs(curvature));
        v = std::clamp(v, min_speed_, max_speed_);

        if (slowdown_distance_ > 1e-6 && remaining < slowdown_distance_)
        {
            double slowdown_ratio = std::clamp(remaining / slowdown_distance_, 0.0, 1.0);
            v = std::min(v, max_speed_ * slowdown_ratio);
        }

        double omega = std::clamp(v * curvature,
                                  -max_angular_velocity_,
                                  max_angular_velocity_);

        publish_ramped_command(v, omega, dt);

        publish_debug(robot, lookahead, idx);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuitNode>());
    rclcpp::shutdown();
}
