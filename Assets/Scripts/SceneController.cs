using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneController : MonoBehaviour
{
    public ImageSynthesis synth;
    public Camera cam;
    //cmd line arguments
    private int numberOfIterations;
    private bool getImage;
    private bool shiftCamera;
    private bool rotateCamera;
    //init parameters
    private Vector3 camInitialPosition = new Vector3(0f, 0f, 15f);
    private Vector3 camInitialRotation = new Vector3(0f, 180f, 0f);
    private Vector3 faceInitialPosition = new Vector3(0f, 0f, 0f);
    private Vector3 faceInitialRotation = new Vector3(270f, 180f, 0f);
    private Vector3 maskInitialPosition = new Vector3(0f, 5.2f, 0f);
    private bool[] valid_stickers;
    private Vector3[] stickers_locs;
    private int frameCounter = 0;
    private int iterationCount = 0;
    private GameObject face;
    private GameObject mask;
    private RotationPaths stage = RotationPaths.up_to_back;
    private float speed = 570;
    private Quaternion startOrientation;
    private int angleAmount;
    private bool doingRotation = false;
    private bool iterationComplete = false;
    
    private string GetArg(string name)
    {
        var args = System.Environment.GetCommandLineArgs();
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == name && args.Length > i + 1)
            {
                return args[i + 1];
            }
        }
        return null;
    }

    void Start()
    {
        var iterationsString = GetArg("-iterations");
        if (!int.TryParse(iterationsString, out numberOfIterations))
        {
            numberOfIterations = 10;
        }
        iterationsString = GetArg("-image");
        if (!bool.TryParse(iterationsString, out getImage))
        {
            getImage = false;
        }
        iterationsString = GetArg("-shift");
        if (!bool.TryParse(iterationsString, out getImage))
        {
            shiftCamera = true;
        }
        iterationsString = GetArg("-rotate");
        if (!bool.TryParse(iterationsString, out getImage))
        {
            rotateCamera = true;
        }
        cam.transform.eulerAngles = camInitialRotation;
        cam.transform.position = camInitialPosition;
        face = GameObject.Find("face");
        mask = GameObject.Find("mask");
        startOrientation = face.transform.rotation;
        face.transform.position = faceInitialPosition;
        face.transform.eulerAngles = faceInitialRotation;
        mask.transform.localPosition = maskInitialPosition;
        float randx = Random.Range(-3f, 3f);
        float randy = Random.Range(-3f, 3f);
        float randz = Random.Range(-3f, 3f);
        mask.transform.localRotation = Quaternion.Euler(randx, randy, randz);
        if (shiftCamera)
        {
            randx = Random.Range(-4f, 4f);
            randy = Random.Range(-4f, 4f);
            randz = Random.Range(12f, 20f);
            cam.transform.position = new Vector3(randx, randy, randz);
        }
        if (rotateCamera)
        {
            randx = Random.Range(-5f, 5f);
            randy = Random.Range(175f, 185f);
            randz = Random.Range(-5f, 5f);
            cam.transform.eulerAngles = new Vector3(randx, randy, randz);
        }

        //numberOfIterations = 5;
    }
    void FixedUpdate()
    {
        //Debug.DrawRay(Vector3.zero, Vector3.forward * 10, Color.cyan, 1);
        //Debug.DrawRay(Vector3.zero, face.transform.forward * 10, Color.blue, 1);
        //Debug.DrawRay(Vector3.zero, face.transform.up * 10, Color.green, 1);
        //Debug.DrawRay(Vector3.zero, face.transform.right * 10, Color.red, 1);
        if (iterationCount < numberOfIterations)
        {
            if (!iterationComplete)
            {
                smoothPath();
            }
            else
            {
                iterationCount++;
                frameCounter = 0;
                iterationComplete = false;
                float randx = Random.Range(-3f, 3f);
                float randy = Random.Range(-3f, 3f);
                float randz = Random.Range(-3f, 3f);
                mask.transform.localRotation = Quaternion.Euler(randx, randy, randz);
                if (shiftCamera)
                {
                    randx = Random.Range(-4f, 4f);
                    randy = Random.Range(-4f, 4f);
                    randz = Random.Range(12f, 20f);
                    cam.transform.position = new Vector3(randx, randy, randz);
                }
                if (rotateCamera)
                {
                    randx = Random.Range(-5f, 5f);
                    randy = Random.Range(175f, 185f);
                    randz = Random.Range(-5f, 5f);
                    cam.transform.eulerAngles = new Vector3(randx, randy, randz);
                }
                stage = RotationPaths.up_to_back;
            }
        }
        else
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
    private IEnumerator Rotate90(Vector3 axis, float angle, bool immediate)
    {
        axis = transform.InverseTransformDirection(axis);
        if (immediate)
        {
            yield return new WaitForEndOfFrame();
            face.transform.rotation = startOrientation * Quaternion.AngleAxis(angle, axis);
        }
        else
        {
            float amount = 0;

            while (amount < angle)
            {
                yield return new WaitForEndOfFrame();
                string filename = $"image_{iterationCount.ToString().PadLeft(5, '0')}_{frameCounter.ToString().PadLeft(5, '0')}";
                synth.Save(filename, 512, 512, "captures", 1, getImage);
                amount += Time.fixedDeltaTime * speed;
                face.transform.rotation = startOrientation * Quaternion.AngleAxis(amount, axis);
                frameCounter++;
            }
            face.transform.rotation = startOrientation * Quaternion.AngleAxis(angle, axis);
        }        
    }
    void smoothPath()
    {
        switch (stage)
        {
            case RotationPaths.up_to_back:
                {
                    if (doingRotation == false)
                    {
                        startOrientation = face.transform.rotation;
                        angleAmount = 90;
                        doingRotation = true;
                        StartCoroutine(Rotate90(Vector3.right, angleAmount, false));
                    }
                    else
                    {
                        if (face.transform.rotation == startOrientation * Quaternion.AngleAxis(angleAmount, Vector3.right))
                        {
                            doingRotation = false;
                            stage = RotationPaths.back_to_back_360;
                        }
                    }
                    break;
                }
            case RotationPaths.back_to_back_360:
                {
                    if (doingRotation == false)
                    {
                        startOrientation = face.transform.rotation;
                        angleAmount = 360;
                        doingRotation = true;
                        StartCoroutine(Rotate90(Vector3.down, angleAmount, false));
                    }
                    else
                    {
                        if (face.transform.rotation == startOrientation * Quaternion.AngleAxis(angleAmount, Vector3.down))
                        {
                            doingRotation = false;
                            stage = RotationPaths.back_to_up;
                        }
                    }
                    break;
                }
            case RotationPaths.back_to_up:
                {
                    if (doingRotation == false)
                    {
                        startOrientation = face.transform.rotation;
                        angleAmount = 90;
                        doingRotation = true;
                        StartCoroutine(Rotate90(Vector3.left, angleAmount, true));
                    }
                    else
                    {
                        if (face.transform.rotation == startOrientation * Quaternion.AngleAxis(angleAmount, Vector3.left))
                        {
                            doingRotation = false;
                            iterationComplete = true;
                        }
                    }
                    break;
                }
        }
        
    }
    enum RotationPaths
    {
        up_to_back = 0,
        back_to_back_360 = 1,
        back_to_up = 2
    };
    void randomRot()
    {
        if (frameCounter < numberOfIterations)
        {
            cam.transform.position = new Vector3(0f, 0f, Random.Range(10f, 30f));
            var z = Random.Range(-15f, 15f);
            var y = Random.Range(0f, 360f);
            float x;
            if ((y >= 90f) && (y <= 270f))
            {
                x = Random.Range(0f, -90f);
            }
            else
            {
                x = Random.Range(0f, 90f);
            }
            GameObject.Find("face").transform.eulerAngles = new Vector3(x, y, z);
            float randx = Random.Range(-3f, 3f);
            float randy = Random.Range(-3f, 3f);
            float randz = Random.Range(-3f, 3f);
            GameObject.Find("mask").transform.localRotation = Quaternion.Euler(randx, randy, randz);
            //GameObject.Find("mask").transform.localScale = new Vector3(Random.Range(0.9f, 1.1f), 1f, Random.Range(0.9f, 1.1f));
            string filename = $"image_{frameCounter.ToString().PadLeft(5, '0')}";
            synth.Save(filename, 512, 512, "captures", 1, getImage);
        }
        else
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
        frameCounter++;
    }
    //bool[] getVisibleStickers()
    //{
    //    bool[] visiblStickers = new bool[9];
    //    var whitePixels = 0;
    //    var blackPixels = 0;
    //    for (int i = 0; i < image.width; i++)
    //        for (int j = 0; j < image.height; j++)
    //        {
    //            Color pixel = image.GetPixel(i, j);

    //            // if it's a white color then just debug...
    //            if (pixel == Color.white)
    //                whitePixels++;
    //            else
    //                blackPixels++;
    //        }
    //    Debug.Log(string.Format("White pixels {0}, black pixels {1}", whitePixels, blackPixels));
    //    return visibleStickers;
    //}
    //void debugRayCast()
    //{
    //    for (int i = 1; i < 10; i++)
    //    {
    //        Vector3 sticker_3dloc = GameObject.Find("Sticker_" + i).transform.position;
    //        Vector3 test = cam.WorldToViewportPoint(sticker_3dloc);
    //        if ((test.z >= 0) && (test.x >= 0) && (test.x <= 1) && (test.y >= 0) && (test.y <= 1))
    //        {
    //            RaycastHit hit;
    //            Vector3 direction = cam.transform.position - sticker_3dloc;
    //            if (Physics.Raycast(sticker_3dloc, direction, out hit, Mathf.Infinity))
    //                if (hit.collider.tag == "MainCamera")
    //                    Debug.Log("camera!" + i);
    //            Debug.DrawRay(sticker_3dloc, direction * 20, Color.yellow);
    //        }
    //    }
    //}
    //void GenerateRandom()
    //{
    //    /*
    //    for (int i = 0; i < maxObjects; i++)
    //    {
    //        if (created[i] != null)
    //        {
    //            Destroy(created[i]);
    //        }
    //    }*/
    //    pool.ReclaimAll();
    //    int objectsThisTime = Random.Range(minObjects, maxObjects);
    //    for (int i = 0; i < objectsThisTime; i++)
    //    {
    //        //pick prefab
    //        int prefabIndex = Random.Range(0, prefabs.Length);
    //        GameObject prefab = prefabs[prefabIndex];
    //        //pos
    //        float newX, newY, newZ;
    //        newX = Random.Range(-10.0f, 10.0f);
    //        newY = Random.Range(10f, 15.0f);
    //        newZ = Random.Range(-10.0f, 10.0f);
    //        Vector3 newPos = new Vector3(newX, newY, newZ);
    //        //rot
    //        var newRot = Random.rotation;
    //        //create
    //        var shape = pool.Get((ShapeLabel)prefabIndex);
    //        var newObj = shape.obj;
    //        newObj.transform.position = newPos;
    //        newObj.transform.rotation = newRot;
    //        //var newObj = Instantiate(prefab, newPos, newRot);
    //        //created[i] = newObj;
    //        //scale
    //        float scaleFactor = Random.Range(0.5f, 3.0f);
    //        Vector3 newScale = new Vector3(scaleFactor, scaleFactor, scaleFactor);
    //        newObj.transform.localScale = newScale;
    //        //color
    //        float newR, newG, newB;
    //        newR = Random.Range(0.0f, 1.0f);
    //        newG = Random.Range(0.0f, 1.0f);
    //        newB = Random.Range(0.0f, 1.0f);
    //        var newColor = new Color(newR, newG, newB);
    //        newObj.GetComponent<Renderer>().material.color = newColor;
    //    }
    //    synth.OnSceneChange();
    //}
    //void saveData(string fileName)
    //{
    //    //valid_stickers = new bool[9];
    //    //stickers_locs = new Vector3[9];
    //    //for (int i = 1; i<10; i++)
    //    //{
    //    //    Vector3 sticker_3dloc = GameObject.Find("Sticker_" + i).transform.position;
    //    //    Vector3 test = cam.WorldToViewportPoint(sticker_3dloc);
    //    //    if ((test.z >= 0) && (test.x >=0) && (test.x <= 1) && (test.y >= 0) && (test.y <= 1)) //object is in viewport
    //    //    {
    //    //        RaycastHit hit;
    //    //        Vector3 direction = cam.transform.position - sticker_3dloc;
    //    //        if (Physics.Raycast(sticker_3dloc, direction, out hit, Mathf.Infinity))
    //    //            if (hit.collider.tag == "MainCamera") 
    //    //                valid_stickers[i - 1] = true;
    //    //            else
    //    //                valid_stickers[i - 1] = false; //hit something else before the camera
    //    //    }
    //    //    else
    //    //    {
    //    //        valid_stickers[i - 1] = false;
    //    //    }
    //    //    Vector3 sticker_2dloc = cam.WorldToScreenPoint(sticker_3dloc);
    //    //    stickers_locs[i - 1] = sticker_2dloc;
    //    //}
    //    //SaveObject obj = new SaveObject { valid_stickers = valid_stickers, stickers_locs = stickers_locs };
    //    //string json = JsonUtility.ToJson(obj);
    //    //Debug.Log(json);
    //    //File.WriteAllText("captures/"+fileName+".json", json);
    //}


}

