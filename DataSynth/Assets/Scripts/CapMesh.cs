using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using System.Linq;

[ExecuteInEditMode]
public class CapMesh : MonoBehaviour
{
    public bool InsertColliders = false;
    public bool DeleteColliders = false;
    public bool ReadSensors = false;
    public bool DeleteSensors = false;
    public bool DisplaySensors = false;
    public bool ProjectSensors = false;
    public Dictionary<string, Vector3> AnchorDictionary = new Dictionary<string, Vector3>();
    public List<Vector3> SensorList = new List<Vector3>();
    public List<Vector3> ProjectedSensorList = new List<Vector3>();

    // Start is called before the first frame update

    // Update is called once per frame
    void Update()
    {
        if (InsertColliders)
        {
            Cloth my_cloth = gameObject.GetComponent<Cloth>();
            GameObject head_mesh = GameObject.Find("head_mesh");
            SphereCollider[] scs = head_mesh.GetComponentsInChildren<SphereCollider>();
            var colliderPairs = new ClothSphereColliderPair[scs.Length];
            for (int i = 0; i < scs.Length; i++)
            {
                ClothSphereColliderPair pair = new ClothSphereColliderPair(scs[i], null);
                colliderPairs[i] = pair;
            }
            my_cloth.sphereColliders = colliderPairs;
            InsertColliders = false;
            Debug.Log("Inserted Colliders");
        }
        if (DeleteColliders)
        {
            Cloth my_cloth = gameObject.GetComponent<Cloth>();
            my_cloth.sphereColliders = null;
            DeleteColliders = false;
            Debug.Log("Deleted Colliders");
        }
        if (ReadSensors)
        {
            readSensorLocations();
            verifySensorPositions();
            normalizeSensorLocations();
            Debug.Log("Read Sensors");
            ReadSensors = false;
        }
        if (DeleteSensors)
        {
            clearSensors();
            Debug.Log("Deleted Sensors");
            DeleteSensors = false;
        }
        if (ProjectSensors)
        {
            projectSensors();
            Debug.Log("Projected Sensors");
            ProjectSensors = false;
        }

    }
    private void clearSensors()
    {
        SensorList.Clear();
        AnchorDictionary.Clear();
        ProjectedSensorList.Clear();
    }
    private void projectSensors()
    {
        ProjectedSensorList.Clear();
        var collider = GetComponent<MeshCollider>();
        foreach (var item in SensorList)
        {
            RaycastHit hit;
            Ray ray = new Ray(item, -item);
            //Debug.DrawRay(item, -item);
            if (!Physics.Raycast(ray, out hit))
            {
                continue;
            }
            Vector3 baryCenter = hit.barycentricCoordinate;
            Vector3 pos = hit.point;
            ProjectedSensorList.Add(pos);
        }
    }
    private void readSensorLocations()
    {
        SensorList.Clear();
        AnchorDictionary.Clear();
        string inputFile = "C:\\src\\UnityCap\\example_models\\example_model.txt";
        string fileData = System.IO.File.ReadAllText(inputFile);
        string[] lines = fileData.Split("\n"[0]);
        for (int i = 0; i < lines.Length; i++)
        {
            string[] lineData = (lines[i].Trim()).Split(" "[0]);
            string name = lineData[0].ToLower();
            if (name != "")
            {
                float x = 0;
                float y = 0;
                float z = 0;
                Assert.IsTrue(float.TryParse(lineData[1], out x));
                Assert.IsTrue(float.TryParse(lineData[2], out y));
                Assert.IsTrue(float.TryParse(lineData[3], out z));
                Vector3 pos = new Vector3(x, y, z);
                int sensor_index = 0;
                bool success = int.TryParse(name, out sensor_index);
                if (success)
                {
                    SensorList.Add(pos);
                }
                else
                {
                    AnchorDictionary.Add(name, pos);
                }
            }
        }
    }

    private bool verifySensorPositions()
    {
        System.Console.WriteLine("Please note: renderer expects input to be in right-handed coordiante system (and internally switches to left-handed).");
        // just some sanity checks on input
        if (AnchorDictionary["lefteye"].x < AnchorDictionary["righteye"].x)
        {
            System.Console.WriteLine("Sanity check failed. Left-Eye sticker x smaller than Right-Eye sticker.");
            return false;
        }
        if (AnchorDictionary["left_triangle"].x < AnchorDictionary["right_triangle"].x)
        {
            System.Console.WriteLine("Sanity check failed. Left-Triangle sticker x smaller than Right-Triangle sticker.");
            return false;
        }
        if (AnchorDictionary["cz"].y != 0f)
        {
            System.Console.WriteLine("Sanity check failed. Top sticker is not centered in y axis.");
            return false;
        }
        if (AnchorDictionary["middle_triangle"].y < AnchorDictionary["lefteye"].y || AnchorDictionary["middle_triangle"].y < AnchorDictionary["righteye"].y)
        {
            System.Console.WriteLine("Sanity check failed. Middle-Triangle sticker is inside of skull.");
            return false;
        }
        if (AnchorDictionary["cz"].z < AnchorDictionary["middle_triangle"].z)
        {
            System.Console.WriteLine("Sanity check failed. Top sticker is not the highest sticker in z direction.");
            return false;
        }
        return true;
    }

    private void normalizeSensorLocations()
    {
        // orient axes
        for (int i = 0; i < SensorList.Count; i++)
        {
            Vector3 new_loc = new Vector3(-SensorList[i].x, SensorList[i].z, SensorList[i].y);
            SensorList[i] = new_loc;
        }
        var keys = new List<string>(AnchorDictionary.Keys);
        foreach (var key in keys)
        {
            Vector3 val = AnchorDictionary[key];
            Vector3 new_loc = new Vector3(-val.x, val.z, val.y);
            AnchorDictionary[key] = new_loc;
        }
        // scale 
        //SkinnedMeshRenderer skin = GetComponent<SkinnedMeshRenderer>();
        //Mesh baked = new Mesh();
        //skin.BakeMesh(baked);
        MeshStudy face_mesh = GameObject.Find("head_mesh").GetComponent<MeshStudy>();
        Vector3 simLeftEye = face_mesh.transform.TransformPoint(face_mesh.originalVertices[face_mesh.selectedIndices[0]]);
        Vector3 simNoseTip = face_mesh.transform.TransformPoint(face_mesh.originalVertices[face_mesh.selectedIndices[1]]);
        Vector3 simRightEye = face_mesh.transform.TransformPoint(face_mesh.originalVertices[face_mesh.selectedIndices[2]]);
        Vector3 simLeftEar = face_mesh.transform.TransformPoint(face_mesh.originalVertices[face_mesh.selectedIndices[3]]);
        Vector3 simRightEar = face_mesh.transform.TransformPoint(face_mesh.originalVertices[face_mesh.selectedIndices[4]]);
        Vector3 simNoseBridge = face_mesh.transform.TransformPoint(face_mesh.originalVertices[face_mesh.selectedIndices[5]]);
        Vector3 simCZ = face_mesh.transform.TransformPoint(face_mesh.originalVertices[face_mesh.selectedIndices[6]]);
        float x1 = simLeftEar.x;
        float x2 = simRightEar.x;
        //float skin_x1 = baked.vertices.Max(v => v.x);
        //float skin_x2 = baked.vertices.Min(v => v.x);
        float anchor_x1 = AnchorDictionary["leftear"].x;
        float anchor_x2 = AnchorDictionary["rightear"].x;
        float xscale = Mathf.Abs(x1 - x2) / Mathf.Abs(anchor_x1 - anchor_x2);

        float y1 = simCZ.y;
        float y2 = simNoseBridge.y;
        float anchor_y1 = AnchorDictionary["top"].y;
        float anchor_y2 = AnchorDictionary["nosebridge"].y;
        float yscale = Mathf.Abs(y1 - y2) / Mathf.Abs(anchor_y1 - anchor_y2);

        float z1 = simCZ.z;
        float z2 = simNoseBridge.z;
        float anchor_z1 = AnchorDictionary["top"].z;
        float anchor_z2 = AnchorDictionary["nosebridge"].z;
        float zscale = Mathf.Abs(z1 - z2) / Mathf.Abs(anchor_z1 - anchor_z2);

        Vector3 scale = new Vector3(xscale, yscale, zscale);

        // calc "center of brain" and center all sensors accordingly
        float my_x = (AnchorDictionary["lefteye"].x + AnchorDictionary["righteye"].x) / 2;
        float my_y = (AnchorDictionary["lefteye"].y + AnchorDictionary["righteye"].y) / 2;
        float my_z = (AnchorDictionary["leftear"].z + AnchorDictionary["rightear"].z) / 2;
        Vector3 pivot_point = new Vector3(my_x, my_y, my_z);

        for (int i = 0; i < SensorList.Count; i++)
        {
            SensorList[i] -= pivot_point;
            SensorList[i] = Vector3.Scale(SensorList[i], scale);
        }
        foreach (var key in keys)
        {
            AnchorDictionary[key] -= pivot_point;
            AnchorDictionary[key] = Vector3.Scale(AnchorDictionary[key], scale);
        }

        pivot_point = AnchorDictionary["nosebridge"] - simNoseBridge;
        for (int i = 0; i < SensorList.Count; i++)
        {
            SensorList[i] -= pivot_point;
        }
        foreach (var key in keys)
        {
            AnchorDictionary[key] -= pivot_point;
        }
    }
}
