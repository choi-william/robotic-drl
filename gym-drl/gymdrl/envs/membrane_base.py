from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)

GRAVITY = -65

##########################
# Exterior Box Dimension #
##########################
BOX_WIDTH = 26.7 
BOX_HEIGHT = 30
BOX_HEIGHT_BELOW_ACTUATORS = 5
EXT_BOX_POLY = [
    (0, BOX_HEIGHT),
    (0, -BOX_HEIGHT_BELOW_ACTUATORS),
    (BOX_WIDTH, -BOX_HEIGHT_BELOW_ACTUATORS),
    (BOX_WIDTH, BOX_HEIGHT)
    ]

###################
# Body Dimensions #
###################
OBJ_SIZE = 4.0 # real dimension
OBJ_POS_OFFSET = 2.0 # real dimension; should be greater than half the object size
ACTUATOR_TIP_SIZE = 1.0 # real dimension 
# Distance between the wall and the center of the first actuator
BOX_SIDE_OFFSET = 1.3 # real dimension
LINK_WIDTH = 6 # real dimension
LINK_HEIGHT = 1.0 # real dimension

# Do not modify
GAP = BOX_WIDTH*(1-BOX_SIDE_OFFSET/BOX_WIDTH*2)/4


####################
# Motor Parameters #
####################
MOTOR_SPEED = 100    # cm/s

###OTHER####
UPPER_TRANSLATION_MIDDLE_JOINT = 10
ACTUATOR_TRANSLATION_MAX = 6.5
ACTUATOR_TRANSLATION_MEAN = ACTUATOR_TRANSLATION_MAX/2
ACTUATOR_TRANSLATION_AMP = ACTUATOR_TRANSLATION_MAX/2


def init_helper(env_obj):
    env_obj.world = b2World(gravity=[0,GRAVITY], doSleep=True)

    env_obj.exterior_box = None
    # Five linear actuators 
    env_obj.actuator_list = []
    # Linkages
    env_obj.link_left_list = [] # four links
    env_obj.link_right_list = [] # four links
    # Drawlist for rendering
    env_obj.drawlist = []

def reset_helper(env_obj):

    # Creating the Exterior Box that defines the 2D Plane
    env_obj.exterior_box = env_obj.world.CreateStaticBody(
        position = (0, 0),
        shapes = b2LoopShape(vertices=EXT_BOX_POLY)
        )
    env_obj.exterior_box.color1 = (0,0,0)
    env_obj.exterior_box.color2 = (0,0,0)

    # Creating 5 actuators 
    actuator_fixture = b2FixtureDef(
        shape = b2CircleShape(radius=ACTUATOR_TIP_SIZE/2),
        density = 1,
        friction = 0.6,
        restitution = 0.0,
        groupIndex = -1
        )

    for i in range(5):
        actuator = env_obj.world.CreateDynamicBody(
            position = (BOX_SIDE_OFFSET+GAP*i,0),
            fixtures = actuator_fixture
            )
        actuator.color1 = (0,0,0.5)
        actuator.color2 = (0,0,0)

        actuator.joint = env_obj.world.CreatePrismaticJoint(
            bodyA = env_obj.exterior_box,
            bodyB = actuator,
            anchor = actuator.position,
            axis = (0,1),
            lowerTranslation = 0,
            upperTranslation = ACTUATOR_TRANSLATION_MAX,
            enableLimit = True,
            maxMotorForce = 10000.0,
            motorSpeed = 0,
            enableMotor = True
            )
        
        env_obj.actuator_list.append(actuator)

    env_obj.drawlist = env_obj.actuator_list

    # Creating the linkages that will form the semi-flexible membrane
    link_fixture = b2FixtureDef(
        shape=b2PolygonShape(box=(LINK_WIDTH/2, LINK_HEIGHT/2)),
        density=1, 
        friction = 0.6,
        restitution = 0.0,
        groupIndex = -1 # neg index to prevent collision
        )

    for i in range(4):
        link_left = env_obj.world.CreateDynamicBody(
            position = (BOX_SIDE_OFFSET+(GAP*i+LINK_WIDTH/2),0),
            fixtures = link_fixture
            )
        link_left.color1 = (0,1,1)
        link_left.color2 = (1,0,1)
        env_obj.link_left_list.append(link_left)
        link_right = env_obj.world.CreateDynamicBody(
            position = (BOX_SIDE_OFFSET+(GAP*(i+1)-LINK_WIDTH/2),0),
            fixtures = link_fixture
            )
        link_right.color1 = (0,1,1)
        link_right.color2 = (1,0,1)
        env_obj.link_right_list.append(link_right)
        
        joint_left = env_obj.world.CreateRevoluteJoint(
            bodyA = env_obj.actuator_list[i],
            bodyB = link_left,
            anchor = env_obj.actuator_list[i].worldCenter,
            collideConnected=False
            )

        joint_right = env_obj.world.CreateRevoluteJoint(
            bodyA = env_obj.actuator_list[i+1],
            bodyB = link_right,
            anchor = env_obj.actuator_list[i+1].worldCenter,
            collideConnected=False
            )

        joint_middle = env_obj.world.CreatePrismaticJoint(
            bodyA = link_left,
            bodyB = link_right,
            anchor = (link_right.position.x-(LINK_WIDTH/2+LINK_HEIGHT/2), (link_left.position.y+link_right.position.y)/2.0),
            axis = (1,0),
            lowerTranslation = 0,
            upperTranslation = UPPER_TRANSLATION_MIDDLE_JOINT,
            enableLimit = True
            )
    # Adding linkages to the drawlist
    env_obj.drawlist = env_obj.link_left_list + env_obj.link_right_list + env_obj.drawlist
    

def destroy_helper(env_obj):
    env_obj.world.DestroyBody(env_obj.exterior_box)
    env_obj.exterior_box = None

    for actuator in env_obj.actuator_list:
        env_obj.world.DestroyBody(actuator)
    env_obj.actuator_list = []

    for left_link in env_obj.link_left_list:
        env_obj.world.DestroyBody(left_link)
    env_obj.link_left_list = []

    for right_link in env_obj.link_right_list:
        env_obj.world.DestroyBody(right_link)
    env_obj.link_right_list = []

def render_helper(env_obj):
    from gym.envs.classic_control import rendering
    # Actuator start position visualized
    env_obj.viewer.draw_polyline( [(0, 0), (BOX_WIDTH, 0)], color=(1,0,1) )

    # Exterior Box Visualized
    box_fixture = env_obj.exterior_box.fixtures[0]
    box_trans = box_fixture.body.transform
    box_path = [box_trans*v for v in box_fixture.shape.vertices]
    box_path.append(box_path[0])
    env_obj.viewer.draw_polyline(box_path, color=env_obj.exterior_box.color2, linewidth=2)

    for obj in env_obj.drawlist:
        for f in obj.fixtures:
            trans = f.body.transform
            if type(f.shape) is b2CircleShape:
                t = rendering.Transform(translation=trans*f.shape.pos)
                env_obj.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                env_obj.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
            else:
                path = [trans*v for v in f.shape.vertices]
                env_obj.viewer.draw_polygon(path, color=obj.color1)
                path.append(path[0])
                env_obj.viewer.draw_polyline(path, color=obj.color2, linewidth=2)