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
from numpy import arctan


def function_M(x, limite):
    if x <= limite:
        return(x)
    elif limite <= x <= 2*limite:
        return(2*limite-x)
    elif 2*limite <= x <= 3*limite:
        return(x-2*limite)
    else:
        return(4*limite-x)


def find_angle(position_toio, theta_toio, position_target):
    theta_toio *= np.pi/180
    # thet_toio en radians
    if position_target[0]-position_toio[0] < 0:
        n = 1
    else:
        n = 0
    if position_target[0]-position_toio[0] == 0:
        if position_target[1]-position_toio[1] > 0:
            theta = np.pi/2
        else:
            theta = 3*np.pi/2
    else:
        theta = arctan(
            (position_target[1]-position_toio[1])/(position_target[0]-position_toio[0]))+n*np.pi
    return(function_M((-theta-theta_toio) % (2*np.pi), np.pi/2))
def distance(position1, position2):
    return(math.sqrt((position1[0]-position2[0])**2+(position1[1]-position2[1])**2))
class ToiofewactionsEnv(gym.Env):
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
        self.action_space = spaces.Discrete(4)
        

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
            theta += 2
            theta = theta % (360)
        
        elif action == 2:
            x += 2*math.cos(float(theta)*math.pi/180)
            y += 2*math.sin(float(theta)*math.pi/180)
        elif action == 3:
            x -= 2*math.cos(float(theta)*math.pi/180)
            y -= 2*math.sin(float(theta)*math.pi/180)
        
        '''
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
        '''
        self.state=x,y,theta,x_target,y_target
        if 0 <= x <= 400 and 0 <= y <= 400:
            done=False
        else:
            done=True
        if not done:
            '''
            try:
                reward=1/abs(state[2]-90)
            except:
                print('test')
                reward=1.0
            '''


            reward = -distance([x,y], [x_target,y_target]) -find_angle(state[0:2],state[2],state[3:5])*180/np.pi
            #reward = (np.pi/(find_angle(state[0:2],state[2],state[3:5])*180))*10**3
            #reward = (1/distance([x,y], [x_target,y_target])+np.pi/(find_angle(state[0:2],state[2],state[3:5])*180))*10**3
        elif self.steps_beyond_done is None:

            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -distance([x,y], [x_target,y_target]) -find_angle(state[0:2],state[2],state[3:5])*180/np.pi
            '''
            try:
                reward=1/abs(state[2]-90)
            except:
                reward=1.0
            '''
           #S reward = (np.pi/(find_angle(state[0:2],state[2],state[3:5])*180))*10**3
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = -1000
       
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state=[random.randrange(380)+10,random.randrange(380)+10,random.randrange(361),random.randrange(380)+10,random.randrange(380)+10]
        #self.state=[random.randrange(380)+10,random.randrange(380)+10,random.randrange(361),random.randrange(380)+10,random.randrange(380)+10]
        self.initial_distance=distance(self.state[0:2],self.state[3:5])
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        rectangle_lenght=15
        toio_size=23
        target_size=20
        x = self.state
        toio_x=x[0]
        toio_y=x[1]
        theta=x[2]*math.pi/180
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
           
          
            
            l,r,t,b = -toio_size/2,toio_size/2,toio_size/2,-toio_size/2
            toio = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.toiotrans = rendering.Transform()
            toio.add_attr(self.toiotrans)
            self.viewer.add_geom(toio)
            
            l,r,t,b = -rectangle_lenght/2,rectangle_lenght/2,rectangle_lenght/2,-rectangle_lenght/2
            rect=rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.recttrans = rendering.Transform()
            #self.recttrans = rendering.Transform(translation=(math.cos(theta)*toio_size/2,math.sin(theta)*toio_size/2 ))
            rect.add_attr(self.toiotrans)
            rect.add_attr(self.recttrans)
            rect.set_color(.5,.5,.8)
            self.viewer.add_geom(rect)
            
            l,r,t,b = -target_size/2,target_size/2,target_size/2,-target_size/2
            rect_target=rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.rect_targettrans = rendering.Transform()
            rect_target.add_attr(self.rect_targettrans)
            #self.axle.add_attr(self.carttrans)
            rect_target.set_color(0,1,0)
            self.viewer.add_geom(rect_target)
          
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
        self.recttrans.set_translation(math.cos(theta)*toio_size/2,math.sin(theta)*toio_size/2 )
        self.toiotrans.set_rotation(theta)
        #self.recttrans.set_rotation(-theta)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None










