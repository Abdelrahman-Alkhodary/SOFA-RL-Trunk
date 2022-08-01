from math import cos
from math import sin
from splib3.numerics import Vec3, Quat
from splib3.animation import animate, AnimationManager
from myController import TrunkController

import os
path = os.path.dirname(os.path.abspath(__file__))+'/mesh/'
dirPath = os.path.dirname(os.path.abspath(__file__))+'/'


class Trunk():
    ''' This prefab is implementing a soft robot inspired by the elephant's trunk.
        The robot is entirely soft and actuated with 8 cables.

        The prefab is composed of:
        - a visual model
        - a collision model
        - a mechanical model for the deformable structure

        The prefab has the following parameters:
        - youngModulus
        - poissonRatio
        - totalMass

        Example of use in a Sofa scene:

        def createScene(root):
            ...
            trunk = Trunk(root)

            ## Direct access to the components
            trunk.displacements = [0., 0., 0., 0., 5., 0., 0., 0.]
    '''

    def __init__(self, parentNode, youngModulus=450, poissonRatio=0.45, totalMass=0.042, inverseMode=False):

        self.inverseMode = inverseMode
        self.node = parentNode.addChild('Trunk')

        self.node.addObject('MeshVTKLoader', name='loader', filename=path+'trunk.vtk')
        self.node.addObject('MeshTopology', src='@loader', name='container')

        self.node.addObject('MechanicalObject', name='dofs', template='Vec3')
        self.node.addObject('UniformMass', totalMass=totalMass)
        self.node.addObject('TetrahedronFEMForceField', template='Vec3', name='FEM', method='large', poissonRatio=poissonRatio,  youngModulus=youngModulus)

        self.__addCables()
        self.addEffectors()

    def __addCables(self):
        length1 = 10.
        length2 = 2.
        lengthTrunk = 195.

        pullPoint = [[0., length1, 0.], [-length1, 0., 0.], [0., -length1, 0.], [length1, 0., 0.]]
        direction = Vec3(0., length2-length1, lengthTrunk)
        direction.normalize()

        nbCables = 4
        for i in range(0, nbCables):
            theta = 1.57*i
            q = Quat(0., 0., sin(theta/2.), cos(theta/2.))

            position = [[0., 0., 0.]]*20
            for k in range(0, 20, 2):
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27)
                position[k+1] = v.rotateFromQuat(q)

            cableL = self.node.addChild('cableL'+str(i))
            cableL.addObject('MechanicalObject', name='dofs',
                                position=pullPoint[i]+[pos.toList() for pos in position])
            cableL.addObject('CableConstraint' if not self.inverseMode else 'CableActuator', template='Vec3', name='cable',
                                hasPullPoint='0',
                                indices=list(range(0, 21)),
                                maxPositiveDisp='70',
                                maxDispVariation='1',
                                minForce=0)
            cableL.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

        for i in range(0, nbCables):
            theta = 1.57*i
            q = Quat(0., 0., sin(theta/2.), cos(theta/2.))

            position = [[0., 0., 0.]]*10
            for k in range(0, 9, 2):
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+21)
                position[k] = v.rotateFromQuat(q)
                v = Vec3(direction[0], direction[1]*17.5*(k/2)+length1, direction[2]*17.5*(k/2)+27)
                position[k+1] = v.rotateFromQuat(q)

            cableS = self.node.addChild('cableS'+str(i))
            cableS.addObject('MechanicalObject', name='dofs',
                                position=pullPoint[i]+[pos.toList() for pos in position])
            cableS.addObject('CableConstraint' if not self.inverseMode else 'CableActuator', template='Vec3', name='cable',
                                hasPullPoint='0',
                                indices=list(range(0, 10)),
                                maxPositiveDisp='40',
                                maxDispVariation='1',
                                minForce=0)
            cableS.addObject('BarycentricMapping', name='mapping',  mapForces=False, mapMasses=False)

    def addVisualModel(self, color=[1., 1., 1., 1.]):
        trunkVisu = self.node.addChild('VisualModel')
        trunkVisu.addObject('MeshSTLLoader', filename=path+'trunk.stl')
        trunkVisu.addObject('OglModel', color=color)
        trunkVisu.addObject('BarycentricMapping')

    def addCollisionModel(self, selfCollision=False):
        trunkColli = self.node.addChild('CollisionModel')
        for i in range(2):
            part = trunkColli.addChild('Part'+str(i+1))
            part.addObject('MeshSTLLoader', name='loader', filename=path+'trunk_colli'+str(i+1)+'.stl')
            part.addObject('MeshTopology', src='@loader')
            part.addObject('MechanicalObject')
            part.addObject('TriangleCollisionModel', group=1 if not selfCollision else i)
            part.addObject('LineCollisionModel', group=1 if not selfCollision else i)
            part.addObject('PointCollisionModel', group=1 if not selfCollision else i)
            part.addObject('BarycentricMapping')

    def fixExtremity(self):
        self.node.addObject('BoxROI', name='boxROI', box=[[-20, -20, 0], [20, 20, 20]], drawBoxes=False)
        self.node.addObject('PartialFixedConstraint', fixedDirections=[1, 1, 1], indices='@boxROI.indices')

    def addEffectors(self, position=[0., 0., 195.]):
        effectors = self.node.addChild('Effectors')
        effectors.addObject('MechanicalObject', position=position, name="myEndEffector")
        #effectors.addObject('PositionEffector', indices=list(range(len(position))), effectorGoal=target)
        effectors.addObject('BarycentricMapping', mapForces=False, mapMasses=False)


