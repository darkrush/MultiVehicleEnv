from typing import *
import numpy as np
import pickle
import time


class VehicleState(object):
    def __init__(self):
        # center point position in x,y axis
        self.coordinate:List[float]= [0.0,0.0]
        # direction of car axis
        self.theta:float = 0.0
        # linear velocity of back point
        self.vel_b:float = 0.0
        # deflection angle of front wheel
        self.phi:float = 0.0
        # Control signal of linear velocity of back point
        self.ctrl_vel_b:float = 0.0
        # Control signal of deflection angle of front wheel
        self.ctrl_phi:float = 0.0
        # cannot move if movable is False. Default is True and be setted as False when crashed.
        self.movable:bool = True
        # Default is False and be setted as True when crashed into other collideable object.
        self.crashed:bool = False

class Vehicle(object):
    def __init__(self):
        # safe radius of the vehicle
        self.r_safe:float = 0.24
        # length of the vehicle
        self.L_car:float = 0.30
        # width of the vehicle
        self.W_car:float = 0.20
        # distance between front and back wheel
        self.L_axis:float = 0.20
        # coefficient of back whell velocity control
        self.K_vel:float = 0.18266
        # coefficient of front wheel deflection control
        self.K_phi:float = 0.298
        # the acceleration of the back whell velocity
        self.dv_dt:float = 2.0
        # the angular acceleration of the back whell velocity
        self.dphi_dt:float = 3.0
        # the color of the vehicle
        self.color:List[float] = [0.0,0.0,0.0]
        # the discrete action table of the vehicle, action_code -> (ctrl_vel,ctrl_phi) 
        self.discrete_table:Dict[int,Tuple[float,float]] = {0:( 0.0, 0.0),
                               1:( 1.0, 0.0), 2:( 1.0, 1.0), 3:( 1.0, -1.0),
                               4:(-1.0, 0.0), 5:(-1.0, 1.0), 6:(-1.0, -1.0)}
        self.data_slot:Dict[str,Any]= {}
        # the state of the vehicle
        self.state:VehicleState = VehicleState()

class EntityState(object):
    def __init__(self):
        # center point position in x,y axis
        self.coordinate:List[float] = [0.0,0.0]

class Entity(object):
    def __init__(self):
        # the redius of the entity
        self.radius:float = 1.0
        # true if the entity can crash into other collideable object
        self.collideable:bool = False
        # the color of the entoty
        self.color:List[float] = [0.0,0.0,0.0]
        # the state of the entity
        self.state:EntityState = EntityState()

# multi-vehicle world
class World(object):
    def __init__(self):
        # list of vehicles and entities (can change at execution-time!)
        self.vehicles:List[Vehicle] = []
        self.landmarks:List[Entity] = []
        self.obstacles:List[Entity] = []
        #range of the main field
        self.field_range:List[float] = [-1.0,-1.0,1.0,1.0]
        self.GUI_port:str = '/dev/shm/gui_port'
        self.GUI_file:Union[BinaryIO,None] = None
        self.sim_state:str = 'init'
        self.real_landmark:int = 0
        

        # simulation timestep
        self.step_t = 1.0
        self.sim_step = 1000
        self.sim_t = self.step_t/self.sim_step
        self.total_time = 0.0

        # the data slot for additional data defined in scenario
        self.data_slot:Dict[str,Any] = {}

    # return all entities in the world
    @property
    def field_center(self):
        return ((self.field_range[0] + self.field_range[2])/2, (self.field_range[1] + self.field_range[3])/2)

    @property
    def field_half_size(self):
        return ((self.field_range[2] - self.field_range[0])/2, (self.field_range[3] - self.field_range[1])/2)

    @property
    def entities(self):
        return self.vehicles + self.landmarks + self.obstacles

    # return all vehicles controllable by external policies
    @property
    def policy_vehicles(self):
        raise NotImplementedError()

    # return all agenvehiclests controlled by world scripts
    @property
    def scripted_vehicles(self):
        raise NotImplementedError()

    def _integrate_state(self):
        for vehicle in self.vehicles:
            state = vehicle.state
            if state.movable:
            #new_state_data = integrate_state_wrap(state, vehicle.L_axis, self.sim_t)
                new_state_data = integrate_state_wrap(state, vehicle, self.sim_t)
                state.coordinate[0], state.coordinate[1], state.theta = new_state_data

    def _check_collision(self):
        for idx_a, vehicle_a in enumerate(self.vehicles):
            if vehicle_a.state.crashed :
                continue
            for idx_b, vehicle_b in enumerate(self.vehicles):
                if idx_a == idx_b:
                    continue
                dist = ((vehicle_a.state.coordinate[0]-vehicle_b.state.coordinate[0])**2
                      +(vehicle_a.state.coordinate[1]-vehicle_b.state.coordinate[1])**2)**0.5
                if dist < vehicle_a.r_safe + vehicle_b.r_safe:
                    vehicle_a.state.collision = True
                    vehicle_a.state.movable = False
                    break

            for obstacle in self.obstacles:
                dist = ((vehicle_a.state.coordinate[0]-obstacle.state.coordinate[0])**2
                      +(vehicle_a.state.coordinate[1]-obstacle.state.coordinate[1])**2)**0.5
                if dist < vehicle_a.r_safe + obstacle.radius:
                    vehicle_a.state.collision = True
                    vehicle_a.state.movable = False
                    break
    
    def dumpGUI(self):
        if self.GUI_port is not None and self.GUI_file is None:
            try:
                self.GUI_file = open(self.GUI_port, "w+b")
            except IOError:
                print('open GUI_file %s failed'%self.GUI_port)
        if self.GUI_port is not None:
            GUI_data = {'field_range':self.field_range,
                        'total_time':self.total_time,
                        'vehicles':self.vehicles,
                        'landmarks':self.landmarks,
                        'obstacles':self.obstacles}
            self.GUI_file.seek(0)
            pickle.dump(GUI_data,self.GUI_file)
            self.GUI_file.flush()

    # update state of the world
    def step(self):
        if self.GUI_port is not None:
            for idx in range(self.sim_step):
                time.sleep(self.sim_t)
                self.total_time += self.sim_t
                self._integrate_state()
                self._check_collision()
                self.dumpGUI()
        else:
            for idx in range(self.sim_step):
                self.total_time += self.sim_t
                self._integrate_state()
                self._check_collision()

def linear_update(x,dx_dt,dt,target):
    if x < target:
        return min(x + dx_dt*dt, target)
    elif x > target:
        return max(x - dx_dt*dt, target)
    return x


def integrate_state_wrap(state:VehicleState, vehicle:Vehicle, dt:float):
    target_vel_b = state.ctrl_vel_b * vehicle.K_vel
    state.vel_b = linear_update(state.vel_b, vehicle.dv_dt, dt, target_vel_b) 
    target_phi = state.ctrl_phi * vehicle.K_phi
    state.phi = linear_update(state.phi, vehicle.dphi_dt, dt, target_phi) 
    
    update_data = integrate_state_njit(state.phi,
                                       state.vel_b,
                                       state.theta,
                                       vehicle.L_axis,
                                       state.coordinate[0],
                                       state.coordinate[1],
                                       dt)
    return update_data


def integrate_state_njit(_phi,_vb,_theta,_L,_x,_y,dt):
    sth = np.sin(_theta)
    cth = np.cos(_theta)
    _xb = _x - cth*_L/2.0
    _yb = _y - sth*_L/2.0
    tphi = np.tan(_phi)
    _omega = _vb/_L*tphi
    _delta_theta = _omega * dt
    if abs(_phi)>0.00001:
        _rb = _L/tphi
        _delta_tao = _rb*(1-np.cos(_delta_theta))
        _delta_yeta = _rb*np.sin(_delta_theta)
    else:
        _delta_tao = _vb*dt*(_delta_theta/2.0)
        _delta_yeta = _vb*dt*(1-_delta_theta**2/6.0)
    _xb += _delta_yeta*cth - _delta_tao*sth
    _yb += _delta_yeta*sth + _delta_tao*cth
    _theta += _delta_theta
    _theta = (_theta/3.1415926)%2*3.1415926

    nx = _xb + np.cos(_theta)*_L/2.0
    ny = _yb + np.sin(_theta)*_L/2.0
    ntheta = _theta
    return nx,ny,ntheta