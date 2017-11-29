#
# BOX2D Test for the Membrane Environment
#

from .framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)

# Exterior Box Dimension
BOX_HEIGHT = 30
BOX_WIDTH = 30
EXT_BOX_POLY = [
    (BOX_WIDTH/15, BOX_HEIGHT),
    (BOX_WIDTH/15, 0),
    (BOX_WIDTH-BOX_WIDTH/15, 0),
    (BOX_WIDTH-BOX_WIDTH/15, BOX_HEIGHT)
    ]

NUM_ACTUATORS = 5

class Membrane(Framework):
    name = "Membrane"
    description = ('This tests the 2D platform with membrane actuator')

    def __init__(self):
        super(Membrane, self).__init__()

        # Creating the Exterior Box that defines the 2D Plane
        self.exterior_box = self.world.CreateStaticBody(
            position = (0, 0),
            shapes = b2LoopShape(vertices=EXT_BOX_POLY)
            )

        # Creating the object to manipulate
        ball_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH/40),
            density = 1,
            friction = 0.6
            )

        self.ball = self.world.CreateDynamicBody(
            position = (BOX_WIDTH/2, BOX_HEIGHT/2),
            fixtures = ball_fixture
            )

        # Creating 5 actuators 
        self.actuator_list = []
        self.actuator_joint_list = []

        actuator_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH/50.0),
            density = 1,
            friction = 0.6,
            groupIndex = -1
            )

        for i in range(5):
            actuator = self.world.CreateDynamicBody(
                position = (BOX_WIDTH/5*i+BOX_WIDTH/10, BOX_HEIGHT/10), 
                fixtures = actuator_fixture
                )

            # actuator = self.world.CreateKinematicBody(
            #     position = (BOX_WIDTH/5*i+BOX_WIDTH/10, BOX_HEIGHT/10), 
            #     shapes = b2CircleShape(radius=BOX_WIDTH/50.0), # diameter is 1/5th of the space allocated for each actuator
            #     )

            actuator_joint = self.world.CreatePrismaticJoint(
                bodyA = self.exterior_box,
                bodyB = actuator,
                anchor = actuator.position,
                axis = (0,1),
                lowerTranslation = 0,
                upperTranslation = BOX_WIDTH*5.0/6.0,
                enableLimit = True,
                maxMotorForce = 1000.0,
                motorSpeed = 0,
                enableMotor = True
                )
            
            self.actuator_list.append(actuator)
            self.actuator_joint_list.append(actuator_joint)

        self.actuator_joint_list[0].motorSpeed = 1.0
        self.actuator_joint_list[3].motorSpeed = 0.5
        self.actuator_joint_list[2].motorSpeed = 0.1

        # Creating the linkages that will form the semi-flexible membrane
        self.link_left_list = [] # four links
        self.link_right_list = [] # four links

        link_fixture = b2FixtureDef(
            shape=b2PolygonShape(box=(BOX_WIDTH/12, BOX_WIDTH/70.0)),
            density=1, 
            friction = 0.6,
            groupIndex = -1 # neg index to prevent collision
            )

        for i in range(1,5):
            link_left = self.world.CreateDynamicBody(
                position = (BOX_WIDTH/5*i-BOX_WIDTH/10+BOX_WIDTH/12,BOX_HEIGHT/10),
                fixtures = link_fixture
                )
            self.link_left_list.append(link_left)

            link_right = self.world.CreateDynamicBody(
                position = (BOX_WIDTH/5*i+BOX_WIDTH/10-BOX_WIDTH/12,BOX_HEIGHT/10),
                fixtures = link_fixture
                )
            self.link_right_list.append(link_right)
            
            joint_left = self.world.CreateRevoluteJoint(
                bodyA = self.actuator_list[i-1],
                bodyB = link_left,
                anchor = self.actuator_list[i-1].worldCenter,
                collideConnected=False
                )

            joint_right = self.world.CreateRevoluteJoint(
                bodyA = self.actuator_list[i],
                bodyB = link_right,
                anchor = self.actuator_list[i].worldCenter,
                collideConnected=False
                )

            joint_middle = self.world.CreatePrismaticJoint(
                bodyA = link_left,
                bodyB = link_right,
                anchor = (link_right.position.x-BOX_WIDTH/12+BOX_WIDTH/70, link_right.position.y),
                axis = (1,0),
                lowerTranslation = 0,
                upperTranslation = BOX_WIDTH/6-BOX_WIDTH/35,
                enableLimit = True
                )


if __name__ == "__main__":
    main(Membrane)
