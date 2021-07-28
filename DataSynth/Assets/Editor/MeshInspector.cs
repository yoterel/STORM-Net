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

[CustomEditor(typeof(MeshStudy))]
public class MeshInspector : Editor
{
    private MeshStudy mesh;
    private Transform handleTransform;
    private Quaternion handleRotation;

    void OnSceneGUI()
    {
        mesh = target as MeshStudy;
        handleTransform = mesh.transform;
        handleRotation = Tools.pivotRotation == PivotRotation.Local ? handleTransform.rotation : Quaternion.identity;

        // ShowHandles on Mesh
        if (mesh.isEditMode)
        {
            if (mesh.originalVertices == null || mesh.normals.Length == 0)
            {
                mesh.Init();
            }
            for (int i = 0; i < mesh.originalVertices.Length; i++)
            {
                ShowHandle(i);
            }
        }

        // Show/ Hide Transform Tool
        switch (mesh.mode)
        {
            case MeshStudy.toolMode.TRANSLATE:
                Tools.current = Tool.Move;
                break;
            case MeshStudy.toolMode.SCALE:
                Tools.current = Tool.Scale;
                break;
            case MeshStudy.toolMode.ROTATE:
                Tools.current = Tool.Rotate;
                break;
            case MeshStudy.toolMode.NONE:
                Tools.current = Tool.None;
                break;
        }
    }

    void ShowHandle(int index)
    {
        Vector3 point = handleTransform.TransformPoint(mesh.originalVertices[index]);
        if (mesh.selectedIndices.Contains(index))
        {
            Handles.color = Color.red;
            if (Handles.Button(point, handleRotation, mesh.pickSize, mesh.pickSize,
                Handles.DotHandleCap)) //1
            {
                mesh.selectedIndices.RemoveAt(mesh.selectedIndices.IndexOf(index));
            }
        }
        else
        {
            Handles.color = Color.blue;
            if (Handles.Button(point, handleRotation, mesh.pickSize, mesh.pickSize,
                Handles.DotHandleCap)) //1
            {
                mesh.selectedIndices.Add(index); //2
            }

        }
    }

    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();
        mesh = target as MeshStudy;
        if (GUILayout.Button("Show Normals"))
        {
            Vector3[] verts = mesh.modifiedVertices.Length == 0 ? mesh.originalVertices : mesh.modifiedVertices;
            Vector3[] normals = mesh.normals;
            Debug.Log(normals.Length);
            for (int i = 0; i < verts.Length; i++)
            {
                Debug.DrawLine(handleTransform.TransformPoint(verts[i]), handleTransform.TransformPoint(verts[i] + normals[i] / 100), Color.green, 1.0f, true);
            }
        }
        if (GUILayout.Button("Clear Selected Vertices"))
        {
            mesh.ClearAllData();
        }
    }
}
