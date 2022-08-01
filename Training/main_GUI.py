from myTrunk import Trunk
from math import cos
from math import sin
import random
from splib3.numerics import Vec3, Quat
from splib3.animation import animate, AnimationManager
from myController import TrunkController
import Sofa
import Sofa.Gui
import Sofa.SofaGL
import SofaRuntime
import Sofa.Simulation
import os
import time
import torch
import numpy as np
import torch.nn as nn
import pandas as pd


# Choose in your script to activate or not the GUI
USE_GUI = True


def main():
    # Make sure to load all SOFA libraries
    SofaRuntime.importPlugin("SofaBaseMechanics")
    SofaRuntime.importPlugin("SofaOpenglVisual")

    #Create the root node
    root = Sofa.Core.Node("root")
    # Call the below 'createScene' function to create the scene graph
    createScene(root)
    root.addObject(TrunkController(node=root))
    Sofa.Simulation.init(root)

    if not USE_GUI:
        for iteration in range(10):
            Sofa.Simulation.animate(root, root.dt.value)
    else:
        # Find out the supported GUIs
        print ("Supported GUIs are: " + Sofa.Gui.GUIManager.ListSupportedGUI(","))
        # Launch the GUI (qt or qglviewer)
        Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
        Sofa.Gui.GUIManager.createGUI(root, __file__)
        Sofa.Gui.GUIManager.SetDimension(1080, 1080)
        # Initialization of the scene will be done here
        Sofa.Gui.GUIManager.MainLoop(root)
        Sofa.Gui.GUIManager.closeGUI()
        print("GUI was closed")

    print("Simulation is done.")


def createScene(rootNode):

    # Choose your resolution mode
    # 1- inverseMode=True, solve the inverse problem and control the end effectors
    # 2- inverseMode=False, solve the direct problem and set the cable displacements by hand
    inverseMode = False

    rootNode.addObject('RequiredPlugin', pluginName=['SoftRobots','SofaSparseSolver','SofaPreconditioner','SofaPython3','SofaConstraint',
                                                     'SofaImplicitOdeSolver','SofaLoader','SofaSimpleFem','SofaBoundaryCondition','SofaEngine',
                                                     'SofaOpenglVisual', 'SofaGeneralLoader'])
    AnimationManager(rootNode)
    rootNode.addObject('VisualStyle', displayFlags='showBehavior')
    rootNode.gravity = [0., -9810., 0.]

    rootNode.addObject('FreeMotionAnimationLoop')
    if inverseMode:
        # For inverse resolution, i.e control of effectors position
        rootNode.addObject('RequiredPlugin', name='SoftRobots.Inverse')
        rootNode.addObject('QPInverseProblemSolver', epsilon=1e-1)
    else:
        # For direct resolution, i.e direct control of the cable displacement
        rootNode.addObject('GenericConstraintSolver', maxIterations=100, tolerance=1e-5)

    simulation = rootNode.addChild('Simulation')

    simulation.addObject('EulerImplicitSolver', name='odesolver', firstOrder=False, rayleighMass=0.1, rayleighStiffness=0.1)
    simulation.addObject('ShewchukPCGLinearSolver', name='linearSolver', iterations=500, tolerance=1.0e-18, preconditioners='precond')
    simulation.addObject('SparseLDLSolver', name='precond')
    simulation.addObject('GenericConstraintCorrection', solverName='precond')

    trunk = Trunk(simulation, inverseMode=inverseMode)
    trunk.addVisualModel(color=[1., 1., 1., 0.8])
    trunk.fixExtremity()

    
    goal_positions_df = pd.read_csv('training_goals.csv')
    _, x, y, z = goal_positions_df.iloc[random.randrange(0,len(goal_positions_df)),:]
    
    goal = rootNode.addChild('Goal')
    goalPosition  = (x, y, z)
    goal.addObject('MechanicalObject', name='goalPos', position=goalPosition, showObject=True, showObjectScale=8, drawMode=2, showColor=[1., 1., 1., 1.])
    goal.addObject('UncoupledConstraintCorrection')
    goal.addObject('EulerImplicitSolver', firstOrder=True)
    goal.addObject('CGLinearSolver')
    
    
    # Use this in direct mode as an example of animation ############
    # def cableanimation(target, factor):
    #     target.cable.value = factor*20
    #
    # animate(cableanimation, {'target': trunk.cableL0}, duration=2, )
    #################################################################

def randomPosition():
    inRange = False
    maxRange = 190 * 190
    while not inRange:
        x = random.randrange(-130, 140)
        y = random.randrange(-150, 120)
        z = random.randrange(0, 190)
        distance = x*x + y*y + z*z
        if distance < maxRange:
            inRange = True
    return [x, y, z]





if __name__ == '__main__':
    main()
