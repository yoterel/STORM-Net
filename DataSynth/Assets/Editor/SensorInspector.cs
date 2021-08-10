/*
 * Copyright (c) 2019 Razeware LLC
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * Notwithstanding the foregoing, you may not use, copy, modify, merge, publish, 
 * distribute, sublicense, create a derivative work, and/or sell copies of the 
 * Software in any work that is designed, intended, or marketed for pedagogical or 
 * instructional purposes related to programming, coding, application development, 
 * or information technology.  Permission for such use, copying, modification,
 * merger, publication, distribution, sublicensing, creation of derivative works, 
 * or sale is expressly withheld.
 *    
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
*/

using UnityEngine;
using UnityEditor;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

[CustomEditor(typeof(CapMesh))]
public class SensorInspector : Editor
{
    private CapMesh mesh;
    private Transform handleTransform;
    private Quaternion handleRotation;

    void OnSceneGUI()
    {
        mesh = target as CapMesh;
        handleTransform = mesh.transform;
        handleRotation = Tools.pivotRotation == PivotRotation.Local ? handleTransform.rotation : Quaternion.identity;
        // ShowHandles on Mesh
        if (mesh.DisplaySensors)
        {
            for (int i = 0; i < mesh.SensorList.Count; i++)
            {
                ShowHandle(mesh.SensorList[i], Color.green);
            }
            foreach (var item in mesh.AnchorDictionary)
            {
                ShowHandle(item.Value, Color.red);
            }
            for (int i = 0; i < mesh.ProjectedSensorList.Count; i++)
            {
                ShowHandle(mesh.ProjectedSensorList[i], Color.blue);
            }
            
        }
    }

    void ShowHandle(Vector3 pos, Color color)
    {
        Handles.color = color;
        Handles.Button(pos, handleRotation, 0.01f, 0.01f,
                Handles.DotHandleCap);
    }

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
    }
}
