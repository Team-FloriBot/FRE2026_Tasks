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
#include <sstream>
#include <iomanip>
#include <vector>
#include <iterator>

class PurePursuitNode : public rclcpp::Node
{
public:
    PurePursuitNode() : Node("pure_pursuit_node")
    {
        this->declare_parameter<double>("lookahead_min", 0.50);
        this->declare_parameter<double>("lookahead_max", 1.00);
        this->declare_parameter<double>("lookahead_gain", 0.25);

        this->declare_parameter<double>("max_speed", 0.22);
        this->declare_parameter<double>("min_speed", 0.08);

        this->declare_parameter<double>("curvature_gain", 4.0);
        this->declare_parameter<double>("pure_pursuit_gain", 0.65);

        this->declare_parameter<double>("max_angular_velocity", 0.65);
        this->declare_parameter<double>("angular_acceleration_limit", 0.60);
        this->declare_parameter<double>("angular_command_filter_alpha", 0.35);

        this->declare_parameter<double>("control_rate", 20.0);
        this->declare_parameter<double>("goal_tolerance", 0.15);
        this->declare_parameter<double>("slowdown_distance", 0.90);

        this->declare_parameter<double>("acceleration_limit", 0.30);
        this->declare_parameter<double>("deceleration_limit", 0.40);

        this->declare_parameter<double>("projection_search_distance", 2.20);
        this->declare_parameter<double>("projection_backtrack_distance", 0.15);

        this->declare_parameter<bool>("heading_filter_enabled", true);
        this->declare_parameter<double>("heading_tolerance", 1.35);

        this->declare_parameter<double>("min_follow_turn_radius", 0.45);

        this->declare_parameter<std::string>("path_topic", "/plan");
        this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
        this->declare_parameter<std::string>("base_link_frame", "base_link");

        lookahead_min_ = get_parameter("lookahead_min").as_double();
        lookahead_max_ = get_parameter("lookahead_max").as_double();
        lookahead_gain_ = get_parameter("lookahead_gain").as_double();

        max_speed_ = get_parameter("max_speed").as_double();
        min_speed_ = get_parameter("min_speed").as_double();

        curvature_gain_ = get_parameter("curvature_gain").as_double();
        pure_pursuit_gain_ = get_parameter("pure_pursuit_gain").as_double();

        max_angular_velocity_ = get_parameter("max_angular_velocity").as_double();
        angular_acceleration_limit_ = get_parameter("angular_acceleration_limit").as_double();

        angular_command_filter_alpha_ = get_parameter("angular_command_filter_alpha").as_double();
        angular_command_filter_alpha_ = std::clamp(angular_command_filter_alpha_, 0.0, 1.0);

        control_rate_ = get_parameter("control_rate").as_double();
        goal_tolerance_ = get_parameter("goal_tolerance").as_double();
        slowdown_distance_ = get_parameter("slowdown_distance").as_double();

        acceleration_limit_ = get_parameter("acceleration_limit").as_double();
        deceleration_limit_ = get_parameter("deceleration_limit").as_double();

        projection_search_distance_ = get_parameter("projection_search_distance").as_double();
        projection_backtrack_distance_ = get_parameter("projection_backtrack_distance").as_double();

        heading_filter_enabled_ = get_parameter("heading_filter_enabled").as_bool();
        heading_tolerance_ = get_parameter("heading_tolerance").as_double();

        min_follow_turn_radius_ = get_parameter("min_follow_turn_radius").as_double();

        path_topic_ = get_parameter("path_topic").as_string();
        cmd_vel_topic_ = get_parameter("cmd_vel_topic").as_string();
        base_link_frame_ = get_parameter("base_link_frame").as_string();

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

    double lookahead_min_;
    double lookahead_max_;
    double lookahead_gain_;

    double max_speed_;
    double min_speed_;

    double curvature_gain_;
    double pure_pursuit_gain_;

    double max_angular_velocity_;
    double angular_acceleration_limit_;
    double angular_command_filter_alpha_;

    double control_rate_;
    double goal_tolerance_;
    double slowdown_distance_;

    double acceleration_limit_;
    double deceleration_limit_;

    double projection_search_distance_;
    double projection_backtrack_distance_;

    double heading_tolerance_;
    double min_follow_turn_radius_;

    bool heading_filter_enabled_;

    double commanded_linear_speed_ = 0.0;
    double commanded_angular_velocity_ = 0.0;

    bool last_control_time_valid_ = false;
    rclcpp::Time last_control_time_;

    std::string path_topic_;
    std::string cmd_vel_topic_;
    std::string base_link_frame_;

    std::vector<double> path_s_;
    double path_length_ = 0.0;
    double progress_s_ = 0.0;

    bool progress_initialized_ = false;
    size_t current_segment_idx_ = 0;

    struct ProjectionResult
    {
        bool valid = false;
        double s = 0.0;
        double distance_sq = std::numeric_limits<double>::max();
        double heading_error = 0.0;
        bool heading_ok = true;
        size_t segment_idx = 0;
        geometry_msgs::msg::Point point;
    };

    double normalize_angle(double angle) const
    {
        constexpr double pi = 3.14159265358979323846;

        while (angle > pi)
            angle -= 2.0 * pi;

        while (angle < -pi)
            angle += 2.0 * pi;

        return angle;
    }

    bool build_path_metrics()
    {
        path_s_.clear();
        path_length_ = 0.0;

        if (path_.poses.size() < 2)
            return false;

        path_s_.resize(path_.poses.size(), 0.0);

        for (size_t i = 0; i < path_.poses.size() - 1; i++)
        {
            const auto & p0 = path_.poses[i].pose.position;
            const auto & p1 = path_.poses[i + 1].pose.position;

            double dx = p1.x - p0.x;
            double dy = p1.y - p0.y;

            path_length_ += std::hypot(dx, dy);
            path_s_[i + 1] = path_length_;
        }

        return path_length_ > 1e-6;
    }

    size_t segment_index_at_s(double s) const
    {
        if (path_.poses.size() < 2 || path_s_.size() != path_.poses.size())
            return 0;

        s = std::clamp(s, 0.0, path_length_);

        auto it = std::upper_bound(path_s_.begin(), path_s_.end(), s);

        size_t idx = 0;

        if (it != path_s_.begin())
            idx = static_cast<size_t>(std::distance(path_s_.begin(), it) - 1);

        idx = std::min(idx, path_.poses.size() - 2);

        while (idx < path_.poses.size() - 2 &&
               path_s_[idx + 1] - path_s_[idx] < 1e-6)
        {
            idx++;
        }

        return idx;
    }

    geometry_msgs::msg::Point point_at_s(double s, size_t * segment_idx = nullptr) const
    {
        geometry_msgs::msg::Point out;

        if (path_.poses.empty())
            return out;

        if (path_.poses.size() < 2 || path_length_ < 1e-6)
            return path_.poses.front().pose.position;

        s = std::clamp(s, 0.0, path_length_);

        if (s >= path_length_ - 1e-9)
        {
            if (segment_idx)
                *segment_idx = path_.poses.size() - 2;

            return path_.poses.back().pose.position;
        }

        size_t idx = segment_index_at_s(s);
        double seg_len = path_s_[idx + 1] - path_s_[idx];

        if (seg_len < 1e-6)
        {
            if (segment_idx)
                *segment_idx = idx;

            return path_.poses[idx].pose.position;
        }

        double t = std::clamp((s - path_s_[idx]) / seg_len, 0.0, 1.0);

        const auto & p0 = path_.poses[idx].pose.position;
        const auto & p1 = path_.poses[idx + 1].pose.position;

        out.x = p0.x + t * (p1.x - p0.x);
        out.y = p0.y + t * (p1.y - p0.y);
        out.z = p0.z + t * (p1.z - p0.z);

        if (segment_idx)
            *segment_idx = idx;

        return out;
    }

    ProjectionResult project_robot_to_path(const geometry_msgs::msg::PoseStamped & robot)
    {
        ProjectionResult best_any;
        ProjectionResult best_heading;

        if (path_.poses.size() < 2 || path_s_.size() != path_.poses.size())
            return best_any;

        double robot_x = robot.pose.position.x;
        double robot_y = robot.pose.position.y;
        double robot_yaw = tf2::getYaw(robot.pose.orientation);

        double start_s = 0.0;
        double end_s = std::min(path_length_, std::max(projection_search_distance_, lookahead_min_));

        if (progress_initialized_)
        {
            start_s = std::max(
                0.0,
                progress_s_ - std::max(0.0, projection_backtrack_distance_));

            end_s = std::min(
                path_length_,
                progress_s_ + std::max(projection_search_distance_, lookahead_min_));
        }

        if (end_s < start_s + 1e-6)
        {
            end_s = std::min(
                path_length_,
                start_s + std::max(projection_search_distance_, lookahead_min_));
        }

        for (size_t i = 0; i < path_.poses.size() - 1; i++)
        {
            double seg_start_s = path_s_[i];
            double seg_end_s = path_s_[i + 1];
            double seg_len = seg_end_s - seg_start_s;

            if (seg_len < 1e-6 || seg_end_s < start_s || seg_start_s > end_s)
                continue;

            const auto & p0 = path_.poses[i].pose.position;
            const auto & p1 = path_.poses[i + 1].pose.position;

            double dx = p1.x - p0.x;
            double dy = p1.y - p0.y;

            double seg_len_sq = dx * dx + dy * dy;

            if (seg_len_sq < 1e-12)
                continue;

            double t_min = std::clamp((start_s - seg_start_s) / seg_len, 0.0, 1.0);
            double t_max = std::clamp((end_s - seg_start_s) / seg_len, 0.0, 1.0);

            if (t_max < t_min)
                continue;

            double raw_t =
                ((robot_x - p0.x) * dx + (robot_y - p0.y) * dy) / seg_len_sq;

            double t = std::clamp(raw_t, t_min, t_max);

            ProjectionResult candidate;
            candidate.valid = true;
            candidate.s = seg_start_s + t * seg_len;
            candidate.segment_idx = i;

            candidate.point.x = p0.x + t * dx;
            candidate.point.y = p0.y + t * dy;
            candidate.point.z = p0.z + t * (p1.z - p0.z);

            double ex = robot_x - candidate.point.x;
            double ey = robot_y - candidate.point.y;

            candidate.distance_sq = ex * ex + ey * ey;

            double segment_yaw = std::atan2(dy, dx);

            candidate.heading_error =
                std::abs(normalize_angle(segment_yaw - robot_yaw));

            candidate.heading_ok =
                !heading_filter_enabled_ ||
                candidate.heading_error <= heading_tolerance_;

            if (!best_any.valid || candidate.distance_sq < best_any.distance_sq)
                best_any = candidate;

            if (candidate.heading_ok &&
                (!best_heading.valid || candidate.distance_sq < best_heading.distance_sq))
            {
                best_heading = candidate;
            }
        }

        if (best_heading.valid)
            return best_heading;

        return best_any;
    }

    void path_callback(const nav_msgs::msg::Path::SharedPtr msg)
    {
        if (msg->poses.size() < 2)
        {
            path_received_ = false;
            tracking_enabled_ = false;
            path_completed_ = true;

            RCLCPP_WARN(
                get_logger(),
                "Empfangener Pfad hat weniger als zwei Posen. Stoppe Path Tracking.");

            return;
        }

        path_ = *msg;

        if (!build_path_metrics())
        {
            path_received_ = false;
            tracking_enabled_ = false;
            path_completed_ = true;

            RCLCPP_WARN(
                get_logger(),
                "Empfangener Pfad hat keine nutzbare Laenge. Stoppe Path Tracking.");

            return;
        }

        path_received_ = true;
        tracking_enabled_ = false;
        path_completed_ = false;

        progress_s_ = 0.0;
        progress_initialized_ = false;
        current_segment_idx_ = 0;

        commanded_linear_speed_ = 0.0;
        commanded_angular_velocity_ = 0.0;
        last_control_time_valid_ = false;

        RCLCPP_INFO(
            get_logger(),
            "Path received: %zu poses, Laenge %.2f m. Path Tracking bereit. Start mit /pure_pursuit_node/set_active.",
            msg->poses.size(),
            path_length_);
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

    bool get_robot_pose(geometry_msgs::msg::PoseStamped & pose)
    {
        try
        {
            auto tf = tf_buffer_->lookupTransform(
                path_.header.frame_id,
                base_link_frame_,
                tf2::TimePointZero);

            pose.header.frame_id = path_.header.frame_id;
            pose.header.stamp = this->now();

            pose.pose.position.x = tf.transform.translation.x;
            pose.pose.position.y = tf.transform.translation.y;
            pose.pose.position.z = tf.transform.translation.z;

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
        double speed_for_lookahead =
            std::max(std::abs(commanded_linear_speed_), min_speed_);

        return std::clamp(
            lookahead_min_ + lookahead_gain_ * speed_for_lookahead,
            lookahead_min_,
            lookahead_max_);
    }

    double ramp_value(
        double current,
        double target,
        double increase_limit,
        double decrease_limit,
        double dt)
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

    void publish_debug(
        const geometry_msgs::msg::PoseStamped & robot,
        const geometry_msgs::msg::Point & projection,
        const geometry_msgs::msg::Point & lookahead,
        size_t segment_idx)
    {
        visualization_msgs::msg::MarkerArray arr;

        visualization_msgs::msg::Marker clear;

        clear.header.frame_id = path_.header.frame_id;
        clear.header.stamp = this->now();
        clear.ns = "pp_debug";
        clear.action = visualization_msgs::msg::Marker::DELETEALL;

        arr.markers.push_back(clear);

        auto sphere = [&](
            int id,
            double x,
            double y,
            float r,
            float g,
            float b,
            double scale)
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

            m.scale.x = scale;
            m.scale.y = scale;
            m.scale.z = scale;

            m.color.a = 1.0;
            m.color.r = r;
            m.color.g = g;
            m.color.b = b;

            return m;
        };

        auto line = [&](
            int id,
            const std::vector<geometry_msgs::msg::Point> & points,
            float r,
            float g,
            float b,
            double width)
        {
            visualization_msgs::msg::Marker m;

            m.header.frame_id = path_.header.frame_id;
            m.header.stamp = this->now();

            m.ns = "pp_debug";
            m.id = id;
            m.type = visualization_msgs::msg::Marker::LINE_STRIP;
            m.action = visualization_msgs::msg::Marker::ADD;

            m.scale.x = width;

            m.color.a = 1.0;
            m.color.r = r;
            m.color.g = g;
            m.color.b = b;

            m.points = points;

            return m;
        };

        auto text = [&](
            int id,
            const geometry_msgs::msg::Point & p,
            const std::string & value)
        {
            visualization_msgs::msg::Marker m;

            m.header.frame_id = path_.header.frame_id;
            m.header.stamp = this->now();

            m.ns = "pp_debug";
            m.id = id;
            m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            m.action = visualization_msgs::msg::Marker::ADD;

            m.pose.position = p;
            m.pose.position.z += 0.35;

            m.scale.z = 0.22;

            m.color.a = 1.0;
            m.color.r = 1.0;
            m.color.g = 1.0;
            m.color.b = 1.0;

            m.text = value;

            return m;
        };

        arr.markers.push_back(
            sphere(
                0,
                robot.pose.position.x,
                robot.pose.position.y,
                1.0,
                0.0,
                0.0,
                0.22));

        arr.markers.push_back(
            sphere(
                1,
                projection.x,
                projection.y,
                1.0,
                0.85,
                0.0,
                0.18));

        arr.markers.push_back(
            sphere(
                2,
                lookahead.x,
                lookahead.y,
                0.0,
                1.0,
                0.0,
                0.22));

        std::vector<geometry_msgs::msg::Point> pursuit_line;

        pursuit_line.push_back(robot.pose.position);
        pursuit_line.push_back(lookahead);

        arr.markers.push_back(
            line(
                3,
                pursuit_line,
                0.0,
                1.0,
                0.0,
                0.035));

        if (path_.poses.size() >= 2)
        {
            segment_idx = std::min(segment_idx, path_.poses.size() - 2);

            std::vector<geometry_msgs::msg::Point> segment_line;

            segment_line.push_back(path_.poses[segment_idx].pose.position);
            segment_line.push_back(path_.poses[segment_idx + 1].pose.position);

            arr.markers.push_back(
                line(
                    4,
                    segment_line,
                    0.0,
                    0.6,
                    1.0,
                    0.06));
        }

        std::ostringstream label;

        label << "seg " << segment_idx
              << "  s " << std::fixed << std::setprecision(2) << progress_s_
              << "/" << path_length_
              << "  v " << commanded_linear_speed_
              << "  w " << commanded_angular_velocity_;

        arr.markers.push_back(text(5, projection, label.str()));

        debug_pub_->publish(arr);
    }

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

        ProjectionResult projection = project_robot_to_path(robot);

        if (!projection.valid)
        {
            RCLCPP_WARN_THROTTLE(
                get_logger(),
                *get_clock(),
                1000,
                "Keine gueltige Pfadprojektion im lokalen Suchfenster. Bremse mit Rampe auf 0.");

            publish_ramped_stop(dt);
            return;
        }

        if (!progress_initialized_)
        {
            progress_s_ = projection.s;
            progress_initialized_ = true;
        }
        else
        {
            progress_s_ = std::max(progress_s_, projection.s);
        }

        progress_s_ = std::clamp(progress_s_, 0.0, path_length_);
        current_segment_idx_ = segment_index_at_s(progress_s_);

        double remaining = std::max(0.0, path_length_ - progress_s_);

        if (remaining <= goal_tolerance_)
        {
            path_completed_ = true;
            tracking_enabled_ = false;

            RCLCPP_INFO(
                get_logger(),
                "Pfadende erreicht. Bremse mit Geschwindigkeitsrampe auf 0.");

            publish_ramped_stop(dt);
            publish_debug(robot, projection.point, path_.poses.back().pose.position, current_segment_idx_);

            return;
        }

        double ld = lookahead_distance();

        double lookahead_s = std::min(path_length_, progress_s_ + ld);

        size_t lookahead_segment_idx = current_segment_idx_;
        auto lookahead = point_at_s(lookahead_s, &lookahead_segment_idx);

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

        if (dist < 0.05)
        {
            publish_ramped_stop(dt);
            return;
        }

        double alpha = std::atan2(y, x);

        double curvature =
            pure_pursuit_gain_ * 2.0 * std::sin(alpha) / dist;

        double v =
            max_speed_ / (1.0 + curvature_gain_ * std::abs(curvature));

        if (slowdown_distance_ > 1e-6 && remaining < slowdown_distance_)
        {
            double slowdown_ratio =
                std::clamp(remaining / slowdown_distance_, 0.0, 1.0);

            v = std::min(v, max_speed_ * slowdown_ratio);
        }

        if (remaining > goal_tolerance_ + 0.05)
            v = std::clamp(v, min_speed_, max_speed_);
        else
            v = std::clamp(v, 0.0, max_speed_);

        double max_omega = max_angular_velocity_;

        if (min_follow_turn_radius_ > 1e-6)
        {
            double radius_limited_omega =
                std::abs(v) / min_follow_turn_radius_;

            max_omega = std::min(max_omega, radius_limited_omega);
        }

        double omega_raw =
            std::clamp(v * curvature, -max_omega, max_omega);

        double omega =
            angular_command_filter_alpha_ * omega_raw +
            (1.0 - angular_command_filter_alpha_) * commanded_angular_velocity_;

        omega = std::clamp(omega, -max_omega, max_omega);

        publish_ramped_command(v, omega, dt);

        publish_debug(robot, projection.point, lookahead, current_segment_idx_);
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PurePursuitNode>());
    rclcpp::shutdown();

    return 0;
}
