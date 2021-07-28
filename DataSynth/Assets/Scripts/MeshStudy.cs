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

using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
[ExecuteInEditMode]
public class MeshStudy : MonoBehaviour
{
    Mesh originalMesh;
    Mesh clonedMesh;

    [HideInInspector]
    public int targetIndex;

    [HideInInspector]
    public Vector3 targetVertex;

    [HideInInspector]
    public Vector3[] originalVertices;

    [HideInInspector]
    public Vector3[] modifiedVertices;

    [HideInInspector]
    public Vector3[] normals;

    [HideInInspector]
    public bool isMeshReady = false;
    public bool isEditMode = true;

    public enum toolMode
    {
        TRANSLATE = 0,
        SCALE = 1,
        ROTATE = 2,
        NONE
    };
    public toolMode mode = toolMode.NONE;
    public List<int> selectedIndices = new List<int>();
    public float pickSize = 0.01f;
    public float radiusOfEffect = 0.3f; //1 
    public float pullValue = 0.3f; //2
    public float duration = 1.2f; //3
    public float rad = 0.5f;
    public int rad_factor = 1;
    public bool CreateColliders = false;
    public bool DeleteColliders = false;
    public bool initialize = false;
    public bool CloneColliders = false;
    public bool my_debug = false;
    private Transform originalMeshTransform;

    void Start()
    {
        Init();
    }

    public void Init()
    {
        MeshFilter meshFilter;
        if (TryGetComponent(out meshFilter))
        {
            isMeshReady = false;
            originalMesh = meshFilter.sharedMesh;
            clonedMesh = new Mesh();
            clonedMesh.name = "clone";
            clonedMesh.vertices = originalMesh.vertices;
            clonedMesh.triangles = originalMesh.triangles;
            clonedMesh.normals = originalMesh.normals;
            meshFilter.mesh = clonedMesh;
            originalVertices = clonedMesh.vertices;
            normals = clonedMesh.normals;
            originalMeshTransform = meshFilter.transform;
        }
        else
        {
            SkinnedMeshRenderer skinned_mesh = GetComponent<SkinnedMeshRenderer>();
            originalMesh = skinned_mesh.sharedMesh;
            clonedMesh = new Mesh();
            skinned_mesh.BakeMesh(clonedMesh, true);
            originalVertices = clonedMesh.vertices;
            normals = clonedMesh.normals;
            originalMeshTransform = skinned_mesh.transform;
        }
        Debug.Log("Init & Cloned");
    }
    public void deleteColliders()
    {
        Debug.Log("Deleting Colliders");
        int children = transform.childCount;
        for (int i = children - 1; i >= 0; i--)
        {
            if (transform.GetChild(i).gameObject.name == "sphere_collider")
            { 
                GameObject.DestroyImmediate(transform.GetChild(i).gameObject);
            }
        }
    }
    public void createColliders()
    {
        deleteColliders();
        Debug.Log("Creating Colliders");
        // selected indices from 0 to 6 are considered landmarks (7 also, but we need a colider on it)
        for (int i = 7; i < selectedIndices.Count; i++)
        {
            GameObject sphere = new GameObject("sphere_collider");
            sphere.transform.SetParent(transform);
            SphereCollider sc = sphere.AddComponent<SphereCollider>();
            sc.radius = rad;
            Vector3 normal = normals[selectedIndices[i]];
            Vector3 worldVertexCorrected = originalMeshTransform.transform.TransformPoint(originalVertices[selectedIndices[i]]) - ((rad / rad_factor) * (normal));
            sphere.transform.position = worldVertexCorrected;
        }
    }

    protected void Update() //1
    {
        //if (this.name == "head_mesh_clone")
        //{
        //    MeshStudy skull = GameObject.Find("head_mesh").GetComponent<MeshStudy>();
        //    selectedIndices = skull.selectedIndices;
        //    CloneColliders = false;
        //}
        if (my_debug)
        {
            SphereCollider[] test = GetComponentsInChildren<SphereCollider>();
            Debug.Log(test[0].transform.position);
            my_debug = false;
        }
        if (initialize)
        {
            Init();
            initialize = false;
        }
        if (CreateColliders)
        {
            createColliders();
            CreateColliders = false;
        }
        if (DeleteColliders)
        {
            deleteColliders();
            DeleteColliders = false;
        }
    }

    public void ClearAllData()
    {
        selectedIndices = new List<int>();
        targetIndex = 0;
        targetVertex = Vector3.zero;
    }
}
