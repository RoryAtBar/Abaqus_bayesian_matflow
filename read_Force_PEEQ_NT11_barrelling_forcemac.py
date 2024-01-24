#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:37:08 2023

@author: w10944rb
"""

from abaqus import *
from  abaqus import session
import job
from odbAccess import *
from odbMaterial import *
from odbSection import *
from abaqusConstants import *
import numpy as np
import __main__
import section
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import optimization
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior
#This version is rewritten simply to read the odb file



def ForceMacro(output_filename,output_directory):
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    session.mdbData.summary()
    o1 = session.openOdb(name=output_directory+output_filename)
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    odb = session.odbs[output_directory+output_filename]
    xy1 = xyPlot.XYDataFromHistory(odb=odb, 
        outputVariableName='CFN2: CFN2     ASSEMBLY__PICKEDSURF36/ASSEMBLY_PART-4-2_PART-4-2-RIGIDSURFACE_ in NSET SAMPLE_TOP', 
        steps=('deform', ), suppressQuery=True, __linkedVpName__='Viewport: 1')
    c1 = session.Curve(xyData=xy1)
    xyp = session.XYPlot('XYPlot-Force')
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    chart.setValues(curvesToPlot=(c1, ), )
    session.charts[chartName].autoColor(lines=True, symbols=True)
    session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
    odb = session.odbs[output_directory+output_filename]
    xy1 = xyPlot.XYDataFromHistory(odb=odb, 
        outputVariableName='CFN2: CFN2     ASSEMBLY__PICKEDSURF79/ASSEMBLY__PICKEDSURF78 in NSET SAMPLE_TOP', 
        steps=('deform', ), suppressQuery=True, __linkedVpName__='Viewport: 1')
    c1 = session.Curve(xyData=xy1)
    xyp = session.xyPlots['XYPlot-Force']
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    chart.setValues(curvesToPlot=(c1, ), )
    session.charts[chartName].autoColor(lines=True, symbols=True)
    x0 = session.xyDataObjects['_temp_2']
    session.writeXYReport(fileName='Force_sample_set1.rpt', xyData=(x0, ))
    odb = session.odbs[output_directory+output_filename]
    xy1 = xyPlot.XYDataFromHistory(odb=odb, 
        outputVariableName='CFN2: CFN2     ASSEMBLY__PICKEDSURF36/ASSEMBLY_PART-4-2_PART-4-2-RIGIDSURFACE_ in NSET SAMPLE_TOP', 
        steps=('deform', ), suppressQuery=True, __linkedVpName__='Viewport: 1')
    c1 = session.Curve(xyData=xy1)
    xyp = session.xyPlots['XYPlot-Force']
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    chart.setValues(curvesToPlot=(c1, ), )
    session.charts[chartName].autoColor(lines=True, symbols=True)
    x0 = session.xyDataObjects['_temp_1']
    session.writeXYReport(fileName='Force_sample_set2.rpt', appendMode=OFF, 
        xyData=(x0, ))

def NT11_plot(output_filename,output_directory):
    session.mdbData.summary()
    o1 = session.openOdb(name=output_directory+output_filename)
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
        variableLabel='NT11', outputPosition=NODAL, )
    session.viewports['Viewport: 1'].odbDisplay.display.setValues(
        plotState=CONTOURS_ON_DEF)
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=19 )
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=19 )
    session.viewports['Viewport: 1'].view.setValues(nearPlane=0.653815, 
        farPlane=0.93521, width=0.0121998, height=0.00814585, 
        viewOffsetX=-0.040211, viewOffsetY=-0.0431251)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=False)
    session.Path(name='Centre_path', type=NODE_LIST, expression=(('SAMPLE', (
        '441:12:-11', 1, )), ))
    pth = session.paths['Centre_path']
    session.XYDataFromPath(name='XYData-Centre_NT11', path=pth, 
        includeIntersections=False, projectOntoMesh=False, 
        pathStyle=PATH_POINTS, numIntervals=10, projectionTolerance=0, 
        shape=DEFORMED, labelType=TRUE_DISTANCE, removeDuplicateXYPairs=True, 
        includeAllElements=False)
    x0 = session.xyDataObjects['XYData-Centre_NT11']
    session.writeXYReport(fileName='NT11.rpt', xyData=(x0, ))


def Barrelling(output_filename,output_directory):
    session.mdbData.summary()
    o1 = session.openOdb(name=output_directory+output_filename)
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    session.viewports['Viewport: 1'].view.setValues(nearPlane=0.650847, 
        farPlane=0.938178, width=0.0274566, height=0.0121673, 
        viewOffsetX=-0.0384923, viewOffsetY=-0.0428389)
    session.Path(name='outer', type=NODE_LIST, expression=(('SAMPLE', (
        '451:11:-11', )), ))
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
        variableLabel='U', outputPosition=NODAL, refinement=(INVARIANT, 
        'Magnitude'))
    xyp = session.XYPlot('XYPlot-1')
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    pth = session.paths['outer']
    xy1 = xyPlot.XYDataFromPath(path=pth, includeIntersections=False, 
        projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
        projectionTolerance=0, shape=DEFORMED, labelType=TRUE_DISTANCE, 
        removeDuplicateXYPairs=True, includeAllElements=False)
    c1 = session.Curve(xyData=xy1)
    chart.setValues(curvesToPlot=(c1, ), )
    session.charts[chartName].autoColor(lines=True, symbols=True)
    session.viewports['Viewport: 1'].setValues(displayedObject=xyp)
    xyp = session.xyPlots['XYPlot-1']
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    pth = session.paths['outer']
    xy1 = xyPlot.XYDataFromPath(path=pth, includeIntersections=False, 
        projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
        projectionTolerance=0, shape=DEFORMED, labelType=Y_COORDINATE, 
        removeDuplicateXYPairs=True, includeAllElements=False)
    c1 = session.Curve(xyData=xy1)
    chart.setValues(curvesToPlot=(c1, ), )
    session.charts[chartName].autoColor(lines=True, symbols=True)
    xyp = session.xyPlots['XYPlot-1']
    chartName = xyp.charts.keys()[0]
    chart = xyp.charts[chartName]
    pth = session.paths['outer']
    xy1 = xyPlot.XYDataFromPath(path=pth, includeIntersections=False, 
        projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
        projectionTolerance=0, shape=DEFORMED, labelType=X_COORDINATE, 
        removeDuplicateXYPairs=True, includeAllElements=False)
    c1 = session.Curve(xyData=xy1)
    chart.setValues(curvesToPlot=(c1, ), )
    session.charts[chartName].autoColor(lines=True, symbols=True)
    pth = session.paths['outer']
    session.XYDataFromPath(name='x-U_data', path=pth, includeIntersections=False, 
        projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
        projectionTolerance=0, shape=DEFORMED, labelType=X_COORDINATE, 
        removeDuplicateXYPairs=True, includeAllElements=False)
    pth = session.paths['outer']
    session.XYDataFromPath(name='X_U-data', path=pth, includeIntersections=False, 
        projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
        projectionTolerance=0, shape=DEFORMED, labelType=X_COORDINATE, 
        removeDuplicateXYPairs=True, includeAllElements=False)
    x0 = session.xyDataObjects['x-U_data']
    session.writeXYReport(fileName='outer_sample_xcoords.rpt', xyData=(x0, ))


def PEEQ_measure(output_filename,output_directory):
    session.mdbData.summary()
    o1 = session.openOdb(
        name= output_directory+output_filename)
    session.viewports['Viewport: 1'].setValues(displayedObject=o1)
    #session.upgradeOdb(
    #    output_directory+output_filename,
    #    output_directory+output_filename,
    #    )
    o7 = session.openOdb(
        output_directory+output_filename)
    session.viewports['Viewport: 1'].setValues(displayedObject=o7)
    session.linkedViewportCommands.setValues(_highlightLinkedViewports=True)
    leaf = dgo.LeafFromPartInstance(partInstanceName=("SAMPLE", ))
    session.viewports['Viewport: 1'].odbDisplay.displayGroup.replace(leaf=leaf)
    session.Path(name='Path-2', type=NODE_LIST, expression=(('SAMPLE', (
        '441:12:-11', 1, )), ))
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=19)
    session.viewports['Viewport: 1'].odbDisplay.setFrame(step=0, frame=19)
    session.viewports['Viewport: 1'].odbDisplay.setPrimaryVariable(
        variableLabel='PEEQ', outputPosition=INTEGRATION_POINT)
    pth = session.paths['Path-2']
    session.XYDataFromPath(name='XYData-2', path=pth, includeIntersections=False, 
        projectOntoMesh=False, pathStyle=PATH_POINTS, numIntervals=10, 
        projectionTolerance=0, shape=UNDEFORMED, labelType=TRUE_DISTANCE)
    x0 = session.xyDataObjects['XYData-2']
    session.writeXYReport(fileName='PEEQ_output.rpt', appendMode=OFF, 
        xyData=(x0, ))

input_file = '~/scratch/Doesitwork.inp'
#output_directory = '~/scratch/'
output_directory = ''
output_filename = 'sub_script_check.odb'

ForceMacro(output_filename, output_directory)
compression_force_file = ''
NT11_plot(output_filename,output_directory)
Barrelling(output_filename,output_directory)
PEEQ_measure(output_filename,output_directory)


