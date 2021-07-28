using UnityEngine;
using System.Collections;
using System.Collections.Generic;

// Fixes wrong Skinned Mesh Renderer bounds occurring with Cloth component.
// Attach this to camera.
//
// Idea: Before camera determines culling, we override the automatically computed
// bounds with our own for all game objects with a Skinned Mesh Render in the present scene.
// In this example, we use the bounds of the undeformed mesh
// scaled by `boundsExtentFactor` in order to take into account possible rescaling
// and rotation, as well as the deformation. A cleaner solution would be to use the
// automatically computed bounds and transform them properly.
[ExecuteInEditMode]
public class FixClothBounds : MonoBehaviour
{

    private List<Bounds> manyBounds;
    private SkinnedMeshRenderer[] skinnedMeshRenderers;
    public float boundsExtentFactor;

    void Start()
    {
        skinnedMeshRenderers = FindObjectsOfType<SkinnedMeshRenderer>();
        manyBounds = new List<Bounds>();
        for (var i = 0; i < skinnedMeshRenderers.Length; i++)
        {
            Bounds bounds = skinnedMeshRenderers[i].sharedMesh.bounds;
            bounds.Expand(bounds.extents * boundsExtentFactor);
            manyBounds.Add(bounds);
        }
    }

    private void OnPreCull()
    {
        FixBounds();
    }

    private void FixBounds()
    {
        for (var i = 0; i < skinnedMeshRenderers.Length; i++)
        {
            skinnedMeshRenderers[i].localBounds = manyBounds[i];
        }
    }

}