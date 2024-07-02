import numpy as np
import math

BURGER_MAX_LIN_VEL = 0.22*0.8 
BURGER_MAX_ANG_VEL = 2.84*0.4 

def trajectory_generation(waypoints, waypoint_dt, output_dt, poly_order=2, space_dim=2):
    
    def fit_spatial_polynomial(waypoints,poly_order, space_dim):
        """
            Fit a spatial polynomial p(s)-> R^space_dim, s in 0~1, to fit the waypoints.
        """
        if waypoints.shape[1]!=space_dim:
            waypoints=waypoints.T

        assert(waypoints.shape[1]==space_dim)

        n = waypoints.shape[0]
        if n<=1:
            return []

        s = np.array([waypoint_dt*i for i in range(n)])
        S = np.vstack([np.power(s,k) for k in range(poly_order+1)])
        S = S.T

        poly_coefs = np.linalg.pinv(S).dot(waypoints)
        return poly_coefs



    def diff_poly_coefs(poly_coefs):
        '''
            Calculate the coefs of the polynomial after taking the first-order derivative.
        '''
        if len(poly_coefs)==1:
            coefs = [0]
        else:
            coefs = np.array(range(len(poly_coefs)))*poly_coefs
            coefs = coefs[1:]
        return coefs

    coef = fit_spatial_polynomial(waypoints,poly_order, space_dim)

    T = waypoints.shape[0]*waypoint_dt
    n_output = int(T/output_dt)

    s = np.array([i*output_dt for i in range(n_output)])
    S = np.vstack([np.power(s,k) for k in range(poly_order+1)])
    S=S.T
    
    dotCoef = np.vstack([diff_poly_coefs(coef[:,i]) for i in range(space_dim)]).T    
    
    p = S[:,:poly_order+1].dot(coef)
    
    pDot = S[:,:poly_order].dot(dotCoef)
        
    theta = np.arctan2(pDot[:,1],pDot[:,0])
    
    return p,theta

def regularize_angle(theta):
    """
        Convert an angle theta to [-pi,pi] representation.
    """
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.angle(cos+sin*1j)
 

class PID(object):
    def __init__(self, Kp=0, Ki=0, Kd=0, max_output=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_output = max_output

        self.prev_error = 0
        self.integral = 0        

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        output = np.clip(output, -self.max_output, self.max_output)
        self.prev_error = error
        return output

class PID_controller(object):
    def __init__(self, pos_PIDs, theta_PIDs, dt):
        
        self.pos_PIDs = pos_PIDs
        self.theta_PIDs = theta_PIDs
        self.dt = dt
        self.prev_error = 0
        self.integral = 0

        self.p_traj = None
        self.theta_traj = None
        self.curr_id = None
        self.n = 0


    def generate_trajectory(self, waypoints):
        self.p_traj = waypoints
        self.curr_id = 0
        self.n = self.p_traj.shape[0]


    def reset(self):    
        self.pos_PIDs.reset()
        self.theta_PIDs.reset()

        self.p_traj = None
        self.theta_traj = None
        self.curr_id = None

    def update(self, loc, yaw):

        if self.p_traj is None:
            return 0.0, 0.0
        

        error = self.p_traj[self.curr_id] - loc

        error_distance = np.linalg.norm(error)
        error_theta =  regularize_angle(regularize_angle(math.atan2(error[1], error[0])) - regularize_angle(yaw))

        print(error_distance, error_theta)

        if error_distance < 0.1:
            self.curr_id += 1
            if self.curr_id >= self.n:
                self.reset()
                return 0.0, 0.0

            error = self.p_traj[self.curr_id] - loc
            error_distance = np.linalg.norm(error)
            error_theta =  regularize_angle(regularize_angle(math.atan2(error[1], error[0])) - regularize_angle(yaw))


        if np.abs(error_theta) >= 0.15:
            omega = self.theta_PIDs.update(error_theta, self.dt)
            v = 0.0
        else:
            omega= self.theta_PIDs.update(error_theta, self.dt)
            v = self.pos_PIDs.update(error_distance, self.dt)

        return v, omega