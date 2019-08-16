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
#import pyglet
from numpy import arctan


MIN_AVOID_DISTANCE = 50
BETA = 100 #gradient amplitude bigger means bigger amplitude
SIGMA = 100 #gradient influence area bigger means bigger area
DELTA = 30 #bias getting number around 0
THETA_DISCOUNT = 5 #limit the influence of the angle bigger means less impact
SPEED  = 124.57 #50% #bigger means faster
SPEED_ANGLE=700.8 #50%
ESPERANCE_TEMPS=21.97
EC_TEMPS=1.004

def gaussian():
    return(np.random.randn() * EC_TEMPS + ESPERANCE_TEMPS)
'''
class DrawText:
    def __init__(self, label:pyglet.text.Label):
        self.label=label
    def render(self):
        self.label.draw()
'''
def function_M(x, limite):
    if x <= limite:
        return(x)
    elif limite <= x <= 2*limite:
        return(2*limite-x)
    elif 2*limite <= x <= 3*limite:
        return(x-2*limite)
    else:
        return(4*limite-x)


def find_angle_to_adjust(position_toio, theta_toio, position_target):
    return(function_M((-find_angle(position_toio, theta_toio, position_target)-theta_toio), np.pi/2))


def find_angle(position_toio, theta_toio, position_target):
    theta_toio *= np.pi/180
    angle = math.atan2(position_target[1]-position_toio[1],(position_target[0]-position_toio[0]))
    if angle<0: #obstacle on the right
        angle += 2*np.pi
    return ((angle-theta_toio) % (2*np.pi))


def distance(position1, position2):
    return(math.sqrt((position1[0]-position2[0])**2+(position1[1]-position2[1])**2))


class ToioTimeObstaclesStateEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.action_space = spaces.Discrete(3)  #CHANGE HERE TO ADD ACTIONS

        self.seed()
        self.viewer = None
        self.reward_label = None
        self.state = None
        self.positions_absolues=None
        self.initial_distance = None
        self.list_of_obstacles = []
        #self.list_of_obstacles = [[random.randrange(400),random.randrange(400),random.randrange(360)] for i in range(3)]
        self.reward = 0 #current reward

    def avoid_collision(self, policy):
        """
            verifies if there is any obstacles in front of the bot, if there is, calculates a angle to turn to to avoid collision.
            It is calculated as addition of the current path vector and a perpendicular of the obstacle vector so the result is a corrected path considering how close the obstacle is.
            @param policy : chosen policy for this turn as relative vector
            @return theta : corrected theta to turn towards
        """
        current_position = [self.state[0],self.state[1]] #absolute positions
        closest_obstacle = None
        obstacles_to_avoid  = [] #list of obstacles?
        corrected_path = [policy[0],policy[1]]
        for obst in self.list_of_obstacles: # absolute 
            if distance(current_position, obst) < MIN_AVOID_DISTANCE:
                obstacles_to_avoid.append(obst)
        for obst in obstacles_to_avoid:
            theta_obstacle = find_angle(current_position, self.positions_absolues[0], obst)*180/np.pi
            if 0<=theta_obstacle<90 or 270<theta_obstacle<=360:
                vector_obstacle = [obst[0]-current_position[0] , obst[1]-current_position[1]]
                #adjust the norm of the vector so that closer means bigger
                vector_obstacle_norm = distance([0,0],vector_obstacle)
                coefficient = (MIN_AVOID_DISTANCE-vector_obstacle_norm)/vector_obstacle_norm #TODO
                vector_obstacle = [vector_obstacle[0]*coefficient , vector_obstacle[1]*coefficient]
                if 0<=theta_obstacle<90:
                    perpendicular_vector = [vector_obstacle[1] , - vector_obstacle[0]]
                else:
                    perpendicular_vector = [-vector_obstacle[1], vector_obstacle[0]]
                corrected_path = [perpendicular_vector[0] + corrected_path[0] , perpendicular_vector[1] + corrected_path[1]]
            else:
                continue
        return math.atan2(corrected_path[1],corrected_path[0])*180/np.pi , distance([0,0], corrected_path)


    def min(obst1, obst2, current_position):
        """
            returns the closest obst from current_position
            @param obst1 : tuple (x,y)
            @param obst2 : tuple (x,y)
            @param current_position : tuple(x,y)
            @return returns either obst1 or obst2
        """
        if obst1 is None:
            return obst2
        elif obst2 is None:
            return obst1
        if distance(obst1, current_position) < distance(obst2, current_position):
            return obst1
        else:
            return obst2

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]



    def step(self, action):
        time_step = gaussian()
        time_step*=10**-3
        assert self.action_space.contains(
            action), "%r (%s) invalid" % (action, type(action))
        x, y, theta_relative, x_relative, y_relative = self.state[0:5]
        theta,x_target,y_target=self.positions_absolues
        if action == 0:
            theta = (theta - SPEED_ANGLE*time_step)%360
            #theta = (theta-20) % 360
            #theta, speed = self.avoid_collision([SPEED*time_step*(np.cos(theta*np.pi/180)), SPEED*time_step*(np.sin(theta*np.pi/180))])
        elif action == 1:
            theta = (theta+SPEED_ANGLE*time_step) % 360
            #theta, speed = self.avoid_collision([SPEED*time_step*(np.cos(theta*np.pi/180)), SPEED*time_step*(np.sin(theta*np.pi/180))])
        elif action == 2:
            x += SPEED*time_step*math.cos(float(theta)*math.pi/180) #x composant of movement vector (absolute coord)
            y += SPEED*time_step*math.sin(float(theta)*math.pi/180)
            #theta, speed = self.avoid_collision([x1, y1])
            #x += SPEED*time_step*math.cos(float(theta)*math.pi/180)
            #y += SPEED*time_step*math.sin(float(theta)*math.pi/180)
        # elif action == 3: #TODO: instead of going backwards it does a 180
        #     x1 = -SPEED*math.cos(float(theta)*math.pi/180)
        #     y1 = -SPEED*math.sin(float(theta)*math.pi/180)
        #     theta, speed = self.avoid_collision([x1, y1])
        #     x -= SPEED*math.cos(float(theta)*math.pi/180)
        #     y -= SPEED*math.sin(float(theta)*math.pi/180)
        theta_relative = find_angle([x, y], theta, [x_target, y_target])*180/np.pi

        dist = distance([x, y], [x_target, y_target])
        list_relative_obstacles=[]
        for obstacle in self.list_of_obstacles:
            dist_relative = distance([x, y], obstacle[0:2])
            theta_relative_obstacle = find_angle([x,y],theta,obstacle[0:2])
            list_relative_obstacles.append(theta_relative_obstacle*180/np.pi)
            list_relative_obstacles.append(-dist_relative*np.sin(theta_relative_obstacle))
            list_relative_obstacles.append(dist_relative*np.cos(theta_relative_obstacle))
        self.state = [x, y, theta_relative, -dist*np.sin(theta_relative/(180/np.pi)), dist*np.cos(theta_relative/(180/np.pi))]+list_relative_obstacles
        self.positions_absolues=[theta,x_target,y_target]
        hit_obstacle = False
        for obstacle in self.list_of_obstacles:
            if distance(obstacle[0:2],[x,y])<=35:
                hit_obstacle = True
        if 0 <= x <= 400 and 0 <= y <= 400 and hit_obstacle == False:
            done = False
            if (x_target-10.0)<=x<=(x_target+10.0) and (y_target-10.0)<=y<=(y_target+10.0):
                print("success!")
                self.reward = 200.0
                done = True
            else:
                theta_reward = theta_relative
                if theta_relative>180:
                    theta_reward -= 360
                theta_reward = -np.abs(theta_reward)
                self.reward = BETA*math.exp(-(dist)**2/(2*SIGMA**2))-DELTA+theta_reward/THETA_DISCOUNT
        else:
            print("doom!")
            done = True
            self.reward = -10000.0
        return np.array(self.state), self.reward, done

    def reset(self):
        list_positions=[]
        [x, y, theta, x_target, y_target] = [random.randrange(380)+10, random.randrange(
            380)+10, random.randrange(361), random.randrange(380)+10, random.randrange(380)+10]
        list_positions.append([x,y])
        list_positions.append([x_target,y_target])
        theta_relative = find_angle([x, y], theta, [x_target, y_target])
        dist = distance([x, y], [x_target, y_target])
        self.list_of_obstacles = []
        for i in range(3):
            position_good = False
            while position_good == False:
                posx,posy = random.randrange(400),random.randrange(400)
                position_good = True
                for pos_registered in list_positions :
                    if distance([posx,posy],pos_registered)<33:
                        position_good = False
            list_positions.append([posx,posy])
            self.list_of_obstacles.append([posx,posy,random.randrange(360)])
        
        #self.list_of_obstacles = [[random.randrange(400),random.randrange(400),random.randrange(360)] for i in range(3)]
        list_relative_obstacles=[]
        for obstacle in self.list_of_obstacles:
            dist_relative = distance([x, y], obstacle[0:2])
            theta_relative_obstacle = find_angle([x,y],theta,obstacle[0:2])
            list_relative_obstacles.append(theta_relative_obstacle*180/np.pi)
            list_relative_obstacles.append(-dist_relative*np.sin(theta_relative_obstacle))
            list_relative_obstacles.append(dist_relative*np.cos(theta_relative_obstacle))
        self.state = [x, y, theta_relative, -dist*np.sin(
            theta_relative), dist*np.cos(theta_relative)]+list_relative_obstacles
        self.positions_absolues=[theta,x_target,y_target]
        self.initial_distance = distance(self.state[0:2], self.state[3:5])
        self.reward = 0

        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400

        rectangle_lenght = 15
        toio_size = 23
        target_size = 20
        x = self.state
        y = self.positions_absolues
        toio_x = x[0]
        toio_y = x[1]
        theta = y[0]*math.pi/180
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.obstacle_trans = []
            self.obstacle_recttrans = []
            self._obstacle_toio_geom = []
            self._obstacle_rect_geom = []

            l, r, t, b = -toio_size/2, toio_size/2, toio_size/2, -toio_size/2
            toio = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.toiotrans = rendering.Transform()
            toio.add_attr(self.toiotrans)
            self.viewer.add_geom(toio)

            l, r, t, b = -rectangle_lenght/2, rectangle_lenght / 2, rectangle_lenght/2, -rectangle_lenght/2
            rect = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.recttrans = rendering.Transform()
            rect.add_attr(self.toiotrans)
            rect.add_attr(self.recttrans)
            rect.set_color(.5, .5, .8)
            self.viewer.add_geom(rect)

            for bot in self.list_of_obstacles:
                l, r, t, b = -toio_size/2, toio_size/2, toio_size/2, -toio_size/2
                obstacle_toio = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.obstacle_trans.append(rendering.Transform())
                obstacle_toio.add_attr(self.obstacle_trans[-1])
                self.viewer.add_geom(obstacle_toio)

                l, r, t, b = -rectangle_lenght/2, rectangle_lenght / 2, rectangle_lenght/2, -rectangle_lenght/2
                obstacle_rect = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.obstacle_recttrans.append(rendering.Transform())
                obstacle_rect.add_attr(self.obstacle_trans[-1])
                obstacle_rect.add_attr(self.obstacle_recttrans[-1])
                obstacle_rect.set_color(.2, .7, .4)
                self.viewer.add_geom(obstacle_rect)

                self._obstacle_toio_geom.append(obstacle_toio)
                self._obstacle_rect_geom.append(obstacle_rect)

            l, r, t, b = -target_size/2, target_size/2, target_size/2, -target_size/2
            rect_target = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.rect_targettrans = rendering.Transform()
            rect_target.add_attr(self.rect_targettrans)
            rect_target.set_color(0, 1, 0)
            self.viewer.add_geom(rect_target)

            self._toio_geom = toio
            self._rect_geom = rect
            self._recttarget_geom = rect_target

            self.reward_label = pyglet.text.Label(text = '00000' , font_size=10, x=10, y=380, color=(0,0,0,255))
            self.viewer.add_geom(DrawText(self.reward_label))

        if self.state is None:
            return None

        toio = self._toio_geom
        l, r, t, b = -toio_size/2, toio_size/2, toio_size/2, -toio_size/2
        toio.v = [(l, b), (l, t), (r, t), (r, b)]

        rect = self._rect_geom
        l, r, t, b = -rectangle_lenght/2, rectangle_lenght / 2, rectangle_lenght/2, -rectangle_lenght/2
        rect.v = [(l, b), (l, t), (r, t), (r, b)]

        for i in range(len(self.list_of_obstacles)):
            toio_o = self._obstacle_toio_geom[i]
            l, r, t, b = -toio_size/2, toio_size/2, toio_size/2, -toio_size/2
            toio_o.v = [(l, b), (l, t), (r, t), (r, b)]

            rect_o = self._obstacle_rect_geom[i]
            l, r, t, b = -rectangle_lenght/2, rectangle_lenght / 2, rectangle_lenght/2, -rectangle_lenght/2
            rect_o.v = [(l, b), (l, t), (r, t), (r, b)]

        rect_target = self._recttarget_geom
        l, r, t, b = -target_size/2, target_size/2, target_size/2, -target_size/2
        rect_target.v = [(l, b), (l, t), (r, t), (r, b)]

        self.rect_targettrans.set_translation(y[1], y[2])
        self.toiotrans.set_translation(toio_x, toio_y)
        self.recttrans.set_translation(math.cos(theta)*toio_size/2, math.sin(theta)*toio_size/2)
        self.toiotrans.set_rotation(theta)
        for i in range(len(self.list_of_obstacles)):
            bot = self.list_of_obstacles[i]
            self.obstacle_trans[i].set_translation(bot[0],bot[1])
            self.obstacle_trans[i].set_rotation(bot[2])
            #self.obstacle_recttrans[i].set_translation(toio_size/2, 0)
            self.obstacle_recttrans[i].set_translation(math.cos(self.list_of_obstacles[i][2])*toio_size/2, math.sin(self.list_of_obstacles[i][2])*toio_size/2)
        self.reward_label.text = self.truncate(self.reward, 5)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def truncate(self, f, n):
        '''Truncates/pads a float f to n decimal places without rounding'''
        s = '{}'.format(f)
        if 'e' in s or 'E' in s:
            return '{0:.{1}f}'.format(f, n)
        i, p, d = s.partition('.')
        return '.'.join([i, (d+'0'*n)[:n]])

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
