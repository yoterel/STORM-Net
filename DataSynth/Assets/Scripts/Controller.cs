using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;
[ExecuteInEditMode]
public class Controller : MonoBehaviour
{
    public bool MoveColliders = false;
    public bool initialize = false;
    public float speed = 0.1f;
    public bool[] animate;
    private Vector3[] targetPositions;
    private SphereCollider[] colliders;
    // Start is called before the first frame update
    void Start()
    {
        Init();
    }
    private void Init()
    {
        colliders = GameObject.Find("head_mesh").GetComponentsInChildren<SphereCollider>();
        animate = new bool[colliders.Length];
        targetPositions = new Vector3[colliders.Length];
        for (int i = 0; i < colliders.Length; i++)
        {
            animate[i] = false;
        }
    }
    // Update is called once per frame
    void Update()
    { 
        if (initialize)
        {
            Init();
            initialize = false;
        }
        if (MoveColliders)
        {
            colliders = GameObject.Find("head_mesh").GetComponentsInChildren<SphereCollider>();
            updateTargetPositions();
            MoveColliders = false;
        }

        for (int i = 0; i < colliders.Length; i++)
        {
            if (animate[i] == true)
            {
                
                float step = speed * Time.deltaTime; // calculate distance to move
                colliders[i].transform.position = Vector3.MoveTowards(colliders[i].transform.position, targetPositions[i], step);

                // Check if the positions are approximately equal.
                if (Vector3.Distance(colliders[i].transform.position, targetPositions[i]) < 0.001f)
                {
                    // stop animation
                    colliders[i].transform.position = targetPositions[i];
                    animate[i] = false;
                }
            }
        }
    }
    private void updateTargetPositions()
    {
        MeshStudy skull = GameObject.Find("head_mesh").GetComponent<MeshStudy>();
        List<int> sel_indices = skull.selectedIndices;
        MeshStudy skull_clone = GameObject.Find("head_mesh_clone").GetComponent<MeshStudy>();
        skull_clone.createColliders();
        SphereCollider[] target_colliders = skull_clone.GetComponentsInChildren<SphereCollider>();
        for (int i = 0; i < colliders.Length; i++)
        {
            //Vector3 world_loc = skull_clone.transform.TransformPoint(skull_clone.originalVertices[sel_indices[i]]);
            Vector3 world_loc = target_colliders[i].transform.position;
            targetPositions[i] = world_loc;
            animate[i] = true;
        }

    }
}
