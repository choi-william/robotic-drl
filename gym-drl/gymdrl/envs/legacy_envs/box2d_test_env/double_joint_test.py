# 
# BOX2D Test for the Double Joint Environment
# The lower left corner of the external box is set to be (0,0)
#

from .framework import (Framework, Keys, main)
from Box2D import (b2CircleShape, b2FixtureDef, b2LoopShape, b2PolygonShape,
                   b2RevoluteJointDef, b2_pi)

import numpy as np
import math

# Exterior Box Dimensions
BOX_HEIGHT = 30
BOX_WIDTH = 30

EXT_BOX_POLY = [
    (0, BOX_HEIGHT),
    (0, 0),
    (BOX_WIDTH, 0),
    (BOX_WIDTH, BOX_HEIGHT)
    ]

# Linkage Dimensions
LINK_LENGTH = 20

# Tip Start Position
TIP_POS = (BOX_WIDTH/4,BOX_HEIGHT/2)

class DoubleJoint (Framework):
    name = "DoubleJoint"
    description = ('This tests the 2D platform with double joint actuator')

    def __init__(self):
        super(DoubleJoint, self).__init__()
        self.world.gravity = (0,0)

        self.exterior_box = None
        self.object = None
        self.first_link = None
        self.second_link = None
        self.actuator_mount = None
        self._create()

    def _create(self):
        # Creating the Exterior Box that defines the 2D Plane
        exterior_box_fixture = b2FixtureDef(
            shape = b2LoopShape(vertices=EXT_BOX_POLY),
            categoryBits = 0x0001,
            maskBits = 0x0004 | 0x0001 | 0x0002
            )
        self.exterior_box = self.world.CreateStaticBody(position = (0, 0))
        self.exterior_box.CreateFixture(exterior_box_fixture)

        # Creating the object to manipulate
        object_fixture = b2FixtureDef(
            shape = b2CircleShape(radius=BOX_WIDTH/40),
            density = 1,
            categoryBits = 0x0002,
            maskBits = 0x0004 | 0x0002 | 0x0001
            )

        self.object = self.world.CreateDynamicBody(
            position = (BOX_WIDTH/2,BOX_HEIGHT/2),
            fixtures = object_fixture,
            linearDamping = 0.3
            )

        # Creating the actuator mount
        self.actuator_mount = self.world.CreateStaticBody(
            position = (-BOX_WIDTH/50,BOX_HEIGHT/2), 
            shapes = b2PolygonShape(box=(BOX_WIDTH/50, BOX_WIDTH/30))
            )

         # Link Position Calculation
        adj = (BOX_WIDTH/50 + TIP_POS[0])/4
        hyp = (LINK_LENGTH/2)
        opp = np.sqrt(hyp*hyp - adj*adj)

        # Creating the actuator
        first_link_fixture = b2FixtureDef(
            shape = b2PolygonShape(box=(LINK_LENGTH/2,BOX_WIDTH/70,(0,0),math.tan(opp/adj))),
            density = 1, 
            categoryBits = 0x0003,
            maskBits = 0
            )

        second_link_fixture = b2FixtureDef(
            shape = b2PolygonShape(box=(LINK_LENGTH/2,BOX_WIDTH/70,(0,0),math.pi-math.tan(opp/adj))),
            density = 1, 
            categoryBits = 0x0003,
            maskBits = 0
            )

        tip_fixture = b2FixtureDef(
            shape = b2CircleShape(pos=(adj,-opp),radius=BOX_WIDTH/50),
            density = 1,
            categoryBits = 0x0004,
            maskBits = 0x0004 | 0x0002 | 0x0001
            )

        self.first_link = self.world.CreateDynamicBody(
            position = self.actuator_mount.position + (adj,opp),
            fixtures = first_link_fixture
            )

        self.first_link.joint = self.world.CreateRevoluteJoint(
            bodyA = self.actuator_mount,
            bodyB = self.first_link,
            anchor = self.actuator_mount.position,
            collideConnected=False,
            maxMotorTorque = 1000.0,
            motorSpeed = 0.0,
            enableMotor = True,
            )

        self.second_link = self.world.CreateDynamicBody(
            position = self.first_link.position + (adj*2,0),
            fixtures = [second_link_fixture, tip_fixture]
            )

        self.second_link.joint = self.world.CreateRevoluteJoint(
            bodyA = self.first_link,
            bodyB = self.second_link,
            anchor = self.first_link.position + (adj,opp),
            collideConnected=False,
            maxMotorTorque = 1000.0,
            motorSpeed = 0.0,
            enableMotor = True,
            )

if __name__ == "__main__":
    main(DoubleJoint)
