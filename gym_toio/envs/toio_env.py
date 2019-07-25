"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
def diff_angle(position_toio,theta_toio,target):
    #theta_toio en degrés
    if target[0]-position_toio[0]<0:
        n=1
    else:
        n=0
    if target[0]-position_toio[0]==0:
        if target[1]-position_toio[1]>0:
            angle=PI/2
        else :
            angle=3*PI/2
    else:
        angle=arctan((target[1]-position_toio[1])/(target[0]-position_toio[0]))+n*PI
    return(-angle-theta_toio)

def distance(position1, position2):
    return(math.sqrt((position1[0]-position2[0])**2+(position1[1]-position2[1])**2))
class ToioEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24 deg        24 deg
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]

    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        '''
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5 # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        '''
        # Angle at which to fail the episode
        '''
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        '''
        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        '''
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        '''
        self.action_space = spaces.Discrete(9)
        

        self.seed()
        self.viewer = None
        self.state = None
        self.initial_distance=None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x,y,theta,x_target,y_target=self.state
        if action == 0:
            theta -= 2
            theta = theta % (360)
        elif action == 1:
            theta -= 2
            theta = theta % (360)
            x += 2*math.cos(float(theta)*math.pi/180)
            y += 2*math.sin(float(theta)*math.pi/180)
        elif action == 2:
            theta -= 2
            theta = theta % (360)
            x -= 2*math.cos(float(theta)*math.pi/180)
            y -= 2*math.sin(float(theta)*math.pi/180)
        elif action == 3:
            theta += 2
            theta = theta % (360)
        elif action == 4:
            theta += 2
            theta = theta % (360)
            x += 2*math.cos(float(theta)*math.pi/180)
            y += 2*math.sin(float(theta)*math.pi/180)
        elif action == 5:
            theta += 2
            theta = theta % (360)
            x -= 2*math.cos(float(theta)*math.pi/180)
            y -= 2*math.sin(float(theta)*math.pi/180)
        elif action == 6:
            x += 2*math.cos(float(theta)*math.pi/180)
            y += 2*math.sin(float(theta)*math.pi/180)
        elif action == 7:
            x -= 2*math.cos(float(theta)*math.pi/180)
            y -= 2*math.sin(float(theta)*math.pi/180)
        self.state=x,y,theta,x_target,y_target
        if 400 <= x <= 800 and 400 <= y <= 800:
            done=False
        else:
            done=True
        if not done:
            reward = (
            1/distance([x,y], [x_target,y_target]))*10**3
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        '''
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x  = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else: # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x  = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x,x_dot,theta,theta_dot)
        done =  x < -self.x_threshold \
                or x > self.x_threshold \
                or theta < -self.theta_threshold_radians \
                or theta > self.theta_threshold_radians
        done = bool(done)
        
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0
        '''
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state=[random.randrange(380)+410,random.randrange(380)+410,random.randrange(361),random.randrange(380)+410,random.randrange(380)+410]
        self.initial_distance=distance(self.state[0:2],self.state[2:4])
        
        #self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 1200
        screen_height = 1200
        '''
        #world_width = self.x_threshold*2
        #scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        #polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0
        '''
        rectangle_lenght=46
        toio_size=46
        target_size=40
        x = self.state
        toio_x=x[0]
        toio_y=x[1]
        theta=x[2]*math.pi/180
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            '''
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            '''
            self.rotation = rendering.Transform()


            self.rect_limit1=rendering.FilledPolygon([(0,0),(1200,0),(1200,400),(0,400)])
            #self.rect_mimit1.add_attr(self.rotation)
            self.rect_limit1.set_color(1,0,0)
            self.viewer.add_geom(self.rect_limit1)

            self.rect_limit2=rendering.FilledPolygon([(0,800),(1200,800),(1200,1200),(0,1200)])
            #self.rect_mimit2.add_attr(self.rotation)
            self.rect_limit2.set_color(1,0,0)
            self.viewer.add_geom(self.rect_limit2)
            
            self.rect_limit3=rendering.FilledPolygon([(0,400),(400,400),(400,800),(0,800)])
            #self.rect_mimit3.add_attr(self.rotation)
            self.rect_limit3.set_color(1,0,0)
            self.viewer.add_geom(self.rect_limit3)

            self.rect_limit4=rendering.FilledPolygon([(800,400),(1200,400),(1200,800),(800,800)])
            #self.rect_mimit4.add_attr(self.rotation)
            self.rect_limit4.set_color(1,0,0)
            self.viewer.add_geom(self.rect_limit4)
            
            l,r,t,b = -toio_size/2,toio_size/2,toio_size/2,-toio_size/2
            toio = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.toiotrans = rendering.Transform()
            #self.toio.add_attr(self.rotation)
            toio.add_attr(self.toiotrans)
            self.viewer.add_geom(toio)
            
            l,r,t,b = -rectangle_lenght/2,rectangle_lenght/2,rectangle_lenght/2,-rectangle_lenght/2
            rect=rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.recttrans = rendering.Transform()
            #self.recttrans = rendering.Transform(translation=(math.cos(theta)*toio_size/2,math.sin(theta)*toio_size/2 ))
            rect.add_attr(self.toiotrans)
            rect.add_attr(self.recttrans)
            #self.rect.add_attr(self.rotation)
            rect.set_color(.5,.5,.8)
            self.viewer.add_geom(rect)
            
            l,r,t,b = -target_size/2,target_size/2,target_size/2,-target_size/2
            rect_target=rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.rect_targettrans = rendering.Transform()
            rect_target.add_attr(self.rect_targettrans)
            #self.rect_target.add_attr(self.rotation)
            #self.axle.add_attr(self.carttrans)
            rect_target.set_color(0,1,0)
            self.viewer.add_geom(rect_target)
            '''
            self.line1=rendering.Line((400,400),(800,400))
            self.line1.set_color(0,0,0)
            self.viewer.add_geom(self.line1)

            self.line2=rendering.Line((800,400),(800,800))
            self.line2.set_color(0,0,0)
            self.viewer.add_geom(self.line2)

            self.line3=rendering.Line((800,800),(400,800))
            self.line3.set_color(0,0,0)
            self.viewer.add_geom(self.line3)

            self.line4=rendering.Line((400,800),(400,400))
            self.line4.set_color(0,0,0)
            self.viewer.add_geom(self.line4)
            '''
          
            '''
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)
            '''
            self._toio_geom = toio
            self._rect_geom = rect
            self._recttarget_geom = rect_target
        if self.state is None: return None

        # Edit the pole polygon vertex
        toio = self._toio_geom
        rect_target=self._recttarget_geom
        l,r,t,b = -toio_size/2,toio_size/2,toio_size/2,-toio_size/2
        toio.v = [(l,b), (l,t), (r,t), (r,b)]
    
        rect = self._rect_geom
        l,r,t,b = -rectangle_lenght/2,rectangle_lenght/2,rectangle_lenght/2,-rectangle_lenght/2
        rect.v = [(l,b), (l,t), (r,t), (r,b)]
        
        rect_target = self._recttarget_geom
        l,r,t,b = -target_size/2,target_size/2,target_size/2,-target_size/2
        rect.v = [(l,b), (l,t), (r,t), (r,b)]
       
        #cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.rect_targettrans.set_translation(x[3],x[4])
        self.toiotrans.set_translation(toio_x, toio_y)
        self.rotation.set_rotation(theta)
        self.recttrans.set_translation(math.cos(theta)*toio_size/2,math.sin(theta)*toio_size/2 )
        self.toiotrans.set_rotation(theta)
        #self.recttrans.set_rotation(-theta)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
