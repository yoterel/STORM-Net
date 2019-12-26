using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneController3 : MonoBehaviour
{
    public ImageSynthesis synth;
    private Camera cam;
    private GameObject camHolder;
    //cmd line arguments
    private int numberOfIterations;
    private bool saveImage;
    private bool saveData;
    private bool shiftCamera;
    private bool rotateCamera;
    //init parameters
    private Vector3 camHolderInitialPosition = new Vector3(0f, 0f, 30f);
    private Vector3 camHolderInitialRotation = new Vector3(0f, 180f, 0f);
    private Vector3 faceInitialPosition = new Vector3(0f, 0f, 0f);
    private Vector3 faceInitialRotation = new Vector3(0f, 0f, 0f);
    private Vector3 maskInitialPosition = new Vector3(0f, 0f, 0f);
    private bool[] valid_stickers;
    private Vector3[] stickers_locs;
    private int frameCounter = 0;
    private int iterationCount = 0;
    private GameObject face;
    private GameObject mask;
    private RotationPaths stage = RotationPaths.front_to_up;
    private float speed = 450;
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
            numberOfIterations = 20;
        }
        iterationsString = GetArg("-save_image");
        if (!bool.TryParse(iterationsString, out saveImage))
        {
            saveImage = true;
        }
        iterationsString = GetArg("-save_data");
        if (!bool.TryParse(iterationsString, out saveData))
        {
            saveData = true;
        }
        iterationsString = GetArg("-shift");
        if (!bool.TryParse(iterationsString, out shiftCamera))
        {
            shiftCamera = true;
        }
        iterationsString = GetArg("-rotate");
        if (!bool.TryParse(iterationsString, out rotateCamera))
        {
            rotateCamera = true;
        }
        face = GameObject.Find("face");
        mask = GameObject.Find("mask");
        camHolder = GameObject.Find("CameraHolder");
        cam = Camera.main;
        startOrientation = face.transform.rotation;
        face.transform.position = faceInitialPosition;
        face.transform.eulerAngles = faceInitialRotation;
        setMaskProperties();
        setCameraProperties();
    }

    void setMaskProperties()
    {
        mask.transform.localPosition = maskInitialPosition;
        //set mask rotation
        float randx = Random.Range(-10f, 10f);
        float randy = Random.Range(-20f, 20f);
        float randz = Random.Range(-15f, 15f);
        mask.transform.localRotation = Quaternion.Euler(randx, randy, randz);
        //set mask scale
        randx = Random.Range(0.8f, 1.2f);
        randy = 1f;
        randz = Random.Range(0.8f, 1.2f);
        mask.transform.localScale = new Vector3(randx, randy, randz);
    }

    void setCameraProperties()
    {
        camHolder.transform.position = camHolderInitialPosition;
        camHolder.transform.eulerAngles = camHolderInitialRotation;
        cam.transform.localPosition = Vector3.zero;
        cam.transform.localEulerAngles = Vector3.zero;
        float randx, randy, randz;
        //set camera properties
        if (shiftCamera)
        { 
            randx = Random.Range(-5f, 5f);
            randy = Random.Range(-4f, 0f);
            randz = Random.Range(18f, 30f);
            camHolder.transform.position = new Vector3(randx, randy, randz);
        }
        if (rotateCamera)
        {
            randx = Random.Range(-5f, 5f);
            randy = Random.Range(175f, 185f);
            randz = Random.Range(-5f, 5f);
            camHolder.transform.eulerAngles = new Vector3(randx, randy, randz);
        }
    }

    void FixedUpdate()
    {
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
                setMaskProperties();
                setCameraProperties();
                stage = RotationPaths.front_to_up;
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

    void smoothPath()
    {
        switch (stage)
        {
            case RotationPaths.front_to_up:
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
                            stage = RotationPaths.up_to_front;
                        }
                    }
                    break;
                }
            case RotationPaths.up_to_front:
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
                synth.Save(filename, 960, 540, "captures", 1, saveImage, saveData);
                amount += Time.fixedDeltaTime * speed;
                face.transform.rotation = startOrientation * Quaternion.AngleAxis(amount, axis);
                float magxyz = 0.2f;
                float x = Random.Range(-1f, 1f) * magxyz;
                float y = Random.Range(-1f, 1f) * magxyz;
                float z = Random.Range(-1f, 1f) * magxyz;
                cam.transform.localPosition = new Vector3(x, y, z);
                float rx = Random.Range(-1f, 1f) * magxyz;
                float ry = Random.Range(-1f, 1f) * magxyz;
                float rz = Random.Range(-1f, 1f) * magxyz;
                cam.transform.localEulerAngles = new Vector3(rx, ry, rz);
                frameCounter++;
            }
            face.transform.rotation = startOrientation * Quaternion.AngleAxis(angle, axis);
        }
    }
    enum RotationPaths
    {
        front_to_up = 0,
        up_to_front = 1
    };
}

