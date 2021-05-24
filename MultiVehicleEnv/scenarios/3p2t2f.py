import numpy as np
from MultiVehicleEnv.basic import World, Vehicle, Entity
from MultiVehicleEnv.scenario import BaseScenario
from MultiVehicleEnv.utils import coord_dist, naive_inference
class Scenario(BaseScenario):
    def make_world(self,args):
        world = World()
        
        self.direction_alpha = args.direction_alpha
        self.add_direction_encoder = args.add_direction_encoder
         
        #for simulate real world
        world.step_t = args.step_t
        world.sim_step = args.sim_step
        world.sim_t = world.step_t/world.sim_step
        world.field_range = [-2.4,-2.4,2.4,2.4]

        # define the task 
        world.data_slot['view_threshold'] = 1.0

        if args.usegui:
            world.GUI_port = args.guiport
        else:
            world.GUI_port = None

        # add agents
        world.vehicles = [Vehicle() for i in range(3)]
        color_list = [[1.0,0.0,0.0],[1.0,0.3,0.3],[1.0,0.6,0.6]]
        for i, agent in enumerate(world.vehicles):
            agent.r_safe     = 0.17
            agent.L_car      = 0.25
            agent.W_car      = 0.18
            agent.L_axis     = 0.2
            agent.K_vel      = 0.18266
            agent.K_phi      = 0.298
            agent.dv_dt      = 2.0
            agent.dphi_dt    = 3.0
            agent.color      = color_list[i]

        # add landmarks
        world.landmarks = []
        for idx in range(2):
            entity = Entity()
            entity.radius = 0.2
            entity.collideable = False
            entity.color  = [0.0,1.0,0.0]
            world.landmarks.append(entity)

        # add obstacles
        world.obstacles = []
        for idx in range(2):
            entity = Entity()
            entity.radius = 0.2
            entity.collideable = True
            entity.color  = [0.0,0.0,0.0]
            world.obstacles.append(entity)

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world:World):

        # set random initial states
        for agent in world.vehicles:
            agent.state.theta = np.random.uniform(0,2*3.14159)
            agent.state.vel_b = 0
            agent.state.phi = 0
            agent.state.ctrl_vel_b = 0
            agent.state.ctrl_phi = 0
            agent.state.movable = True
            agent.state.crashed = False

        conflict = True
        while conflict:
            conflict = False
            all_circle = []
            for landmark in world.landmarks:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-0.5,+0.5)
                    norm_pos = norm_pos + (0.5 if norm_pos>0 else -0.5)
                    landmark.state.coordinate[idx] = norm_pos * scale + trans
                all_circle.append((landmark.state.coordinate[0],landmark.state.coordinate[1],landmark.radius))
    
            for obstacle in world.obstacles:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    obstacle.state.coordinate[idx] = norm_pos * scale*0.5 + trans
                all_circle.append((obstacle.state.coordinate[0],obstacle.state.coordinate[1],obstacle.radius))
    
            for agent in world.vehicles:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    agent.state.coordinate[idx] = norm_pos * scale + trans
                all_circle.append((agent.state.coordinate[0],agent.state.coordinate[1],agent.r_safe))
            
            for idx_a in range(len(all_circle)):
                for idx_b in range(idx_a+1,len(all_circle)):
                    x_a = all_circle[idx_a][0]
                    y_a = all_circle[idx_a][1]
                    r_a = all_circle[idx_a][2]
                    x_b = all_circle[idx_b][0]
                    y_b = all_circle[idx_b][1]
                    r_b = all_circle[idx_b][2]
                    dis = ((x_a - x_b)**2 + (y_a - y_b)**2)**0.5
                    if dis < r_a + r_b:
                        conflict = True
                        break
                if conflict:
                    break
        

        world.real_landmark = np.random.randint(len(world.landmarks))
        real_landmark = world.landmarks[world.real_landmark]
        onehot = np.eye(4)
        def encode_direction(direction):
            if direction[0] > 0 and direction[1] > 0:
                return 0
            if direction[0] < 0 and direction[1] > 0:
                return 1
            if direction[0] < 0 and direction[1] < 0:
                return 2
            if direction[0] > 0 and direction[1] < 0:
                return 3
        
        for agent in world.vehicles:
            direction = [real_landmark.state.coordinate[0] - agent.state.coordinate[0],
                         real_landmark.state.coordinate[1] - agent.state.coordinate[1]]
            agent.data_slot['direction_encoder'] = onehot[encode_direction(direction),:]
                        
    def reward(self, agent:Vehicle, world:World):
                # Adversaries are rewarded for collisions with agents
        rew:float = 0.0

        # direction reward
        prefer_action = naive_inference(agent.state.coordinate[0],
                                        agent.state.coordinate[1],
                                        agent.state.theta)
        same_direction = np.sign(agent.state.ctrl_vel_b) == prefer_action[0] and np.sign(agent.state.ctrl_phi) == prefer_action[1]
        if same_direction:
            rew += self.direction_alpha * 1.0


        # reach reward
        Allreach = True
        real_landmark = world.landmarks[world.real_landmark]
        for agent_a in world.vehicles:
            dist = coord_dist(agent_a.state.coordinate, real_landmark.state.coordinate)
            if dist > agent_a.r_safe +real_landmark.radius:
                Allreach = False
        if Allreach:
            rew += 1.0
            
        # collision reward
        if agent.state.crashed:
            rew -= 1.0

        return rew

    def observation(self,agent:Vehicle, world:World):
        def get_pos(obj , main_agent:Vehicle = None):
            main_shift_x = 0.0 if main_agent  is None else main_agent.state.coordinate[0]
            main_shift_y = 0.0 if main_agent  is None else main_agent.state.coordinate[1]
            x = obj.state.coordinate[0] - main_shift_x 
            y = obj.state.coordinate[1] - main_shift_y

            if isinstance(obj,Vehicle):
                ctheta = np.cos(obj.state.theta)
                stheta = np.sin(obj.state.theta)
                return [x, y, ctheta, stheta]
            elif isinstance(obj,Entity):
                return [x, y]
            else:
                raise TypeError


        # get positions of all obstacles in this agent's reference frame
        obstacle_pos = []
        for obstacle in world.obstacles:
            epos = np.array([obstacle.state.coordinate[0] - agent.state.coordinate[0],
                             obstacle.state.coordinate[1] - agent.state.coordinate[1]])
            obstacle_pos.append(epos)

        agent_pos = [get_pos(agent)]

        # check in view
        in_view = 0.0
        landmark_pos = []
        for landmark in world.landmarks:
            dist = coord_dist(agent.state.coordinate, landmark.state.coordinate)
            if dist < world.data_slot['view_threshold']:
                in_view = 1.0
                landmark_pos.append(get_pos(landmark, agent))
            else:
                landmark_pos.append([0,0])

        # communication of all other agents
        other_pos = []
      
        for other in world.vehicles:
            if other is agent: continue
            other_pos.append(get_pos(other,agent))


        if  self.add_direction_encoder:
            return np.concatenate([agent.data_slot['direction_encoder']] + agent_pos + landmark_pos + obstacle_pos + other_pos + [in_view])
        else:
            return np.concatenate(agent_pos + landmark_pos + obstacle_pos + other_pos)



    def info(self, agent:Vehicle, world:World):
        agent_info:dict = {}
        return agent_info