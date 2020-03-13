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
    private Vector3 camHolderInitialPosition = new Vector3(0f, 25f, 0);
    private Vector3 targetCamHolderPosition;
    private Vector3 camHolderInitialRotation = new Vector3(90f, 0f, 0f);
    private Vector3 faceInitialPosition = new Vector3(0f, 0f, 0f);
    private Vector3 faceInitialRotation = new Vector3(0f, 0f, 0f);
    private Vector3 maskInitialPosition = new Vector3(0f, 0f, 0f);
    private Vector3[] initial_sticker_positions = new Vector3[7];
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
        string[] names = new string[] { "AL", "NZ", "AR", "FP1", "FPZ", "FP2", "CZ"};
        Vector3[] positions = new Vector3[] {
            new Vector3 {x = 2.7f, y = 6f, z = 0f },
            new Vector3 {x = 0f, y = 7.21f, z = -1.74f },
            new Vector3 {x = -2.7f, y = 6f, z = 0f },
            new Vector3 {x = 3f, y = 8f, z = 3.13f },
            new Vector3 {x = 0f, y = 8f, z = 5f },
            new Vector3 {x = -3f, y = 8f, z = 3.13f },
            new Vector3 {x = 0f, y = 0f, z = 10f }};
        for (int i = 0; i < names.Length; i++)
        {
            GameObject sticker = GameObject.Find(names[i]);
            sticker.transform.position = positions[i];
            initial_sticker_positions[i] = sticker.transform.position;
        }
        startOrientation = face.transform.rotation;
        face.transform.position = faceInitialPosition;
        face.transform.eulerAngles = faceInitialRotation;
        setFaceStickersProperties();
        setMaskProperties();
        setCameraProperties();
    }
    void setCapStickersProperties(float mag)
    {
        string[] names = new string[] {"FP1", "FPZ", "FP2", "CZ"};
        for (int i = 0; i < names.Length; i++)
        {
            GameObject sticker = GameObject.Find(names[i]);
            sticker.transform.position = initial_sticker_positions[i];
        }
        //string[] mask_stickers_names = new string[] { "FP1", "FPZ", "FP2", "CZ" };
        //for (int i = 0; i < 4; i++)
        //{
        //    Vector2 my_random_vector= Random.insideUnitCircle * mag;
        //    GameObject sticker = GameObject.Find(mask_stickers_names[i]);
        //    sticker.transform.position += sticker.transform.TransformDirection(Vector3.right) * my_random_vector.x;
        //    sticker.transform.position += sticker.transform.TransformDirection(Vector3.forward) * my_random_vector.y;
        //}
    }
    void setFaceStickersProperties()
    {
        string[] names = new string[] {"AL", "NZ", "AR"};
        for (int i = 0; i < names.Length; i++)
        {
            GameObject sticker = GameObject.Find(names[i]);
            sticker.transform.position = initial_sticker_positions[i];
        }
        //set eye distance
        float eye_dist = Random.Range(3.8f, 5.4f);
        //float eye_dist = Random.Range(2f, 8f);
        GameObject AR = GameObject.Find("AR");
        GameObject AL = GameObject.Find("AL");
        AR.transform.position = new Vector3(-eye_dist / 2, AR.transform.position.y, AR.transform.position.z);
        AL.transform.position = new Vector3(eye_dist / 2, AL.transform.position.y, AL.transform.position.z);
        //set nose distance
        float nose_drop = Random.Range(0f, 0.5f);
        float nose_depth = Random.Range(-0.5f, 0.5f);
        GameObject NZ = GameObject.Find("NZ");
        NZ.transform.position += NZ.transform.TransformDirection(Vector3.forward) * nose_drop;
        NZ.transform.position += NZ.transform.TransformDirection(Vector3.up) * nose_depth;
    }
    void setMaskProperties()
    {
        mask.transform.position = maskInitialPosition;
        //set mask scale
        float randxscale = Random.Range(1f, 1.5f);
        float randyscale = Random.Range(0.8f, 1f);
        float randzscale = Random.Range(1f, 1.2f);
        mask.transform.localScale = new Vector3(randxscale, randyscale, randzscale);
        //set mask rotation
        float randx = Random.Range(-10f, 10f);
        float randy = Random.Range(-15f, 15f);
        float randz = Random.Range(-20f, 20f);
        mask.transform.rotation = Quaternion.Euler(randx, randy, randz);
    }

    float map_range(float s_range_low, float s_range_high, float d_range_low, float d_range_high, float value)
    {
        float m = (d_range_high - d_range_low) / (s_range_high - s_range_low); // slope
        float b = d_range_low - (m * s_range_low);
        float my_range = m * value + b;
        return my_range;
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
            randy = Random.Range(20f, 45f);
            //float rangex = map_range(20f, 35f, 2f, 9f, randy);
            //randx = Random.Range(-rangex, rangex);
            //float rangez = map_range(20f, 35f, 2f, 10f, randy);
            //randz = Random.Range(-rangez, rangez);
            camHolder.transform.position = new Vector3(0, randy, 0);
        }
        if (rotateCamera)
        {
            //randx = Random.Range(-5f, 5f);
            randy = Random.Range(-5f, -5f);
            randz = Random.Range(-5f, 5f);
            cam.transform.localEulerAngles = new Vector3(0f, randy, randz);
        }
        float rand = Random.Range(20f, 45f);
        targetCamHolderPosition = camHolder.transform.position;
        targetCamHolderPosition.y = rand;
    }

    void FixedUpdate()
    {
        if (iterationCount < numberOfIterations)
        {
            if (!iterationComplete)
            {
                smoothPath();
                camPath();
            }
            else
            {
                iterationCount++;
                frameCounter = 0;
                iterationComplete = false;
                //setCapStickersProperties(0.5f);
                setFaceStickersProperties();
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
    void camPath()
    {
        Vector3 dirNormalized = (targetCamHolderPosition - camHolder.transform.position).normalized;
        camHolder.transform.position = camHolder.transform.position + dirNormalized * Time.deltaTime * 5; //magic number for speed
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
                        StartCoroutine(RotateBy(Vector3.left, angleAmount, false));
                    }
                    else
                    {
                        if (face.transform.rotation == startOrientation * Quaternion.AngleAxis(angleAmount, Vector3.left))
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
                        StartCoroutine(RotateBy(Vector3.right, angleAmount, true));
                    }
                    else
                    {
                        if (face.transform.rotation == startOrientation * Quaternion.AngleAxis(angleAmount, Vector3.right))
                        {
                            doingRotation = false;
                            iterationComplete = true;
                        }
                    }
                    break;
                }
        }
        
    }

    private IEnumerator RotateBy(Vector3 axis, float angle, bool immediate)
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
                string filename = $"image_{iterationCount.ToString().PadLeft(6, '0')}_{frameCounter.ToString().PadLeft(3, '0')}";
                synth.Save(filename, 960, 540, "captures", 1, saveImage, saveData);
                amount += Time.fixedDeltaTime * speed;
                face.transform.rotation = startOrientation * Quaternion.AngleAxis(amount, axis);
                float magxyz = 2f;
                //float x = Random.Range(-1f, 1f) * magxyz;
                //float y = Random.Range(-1f, 1f) * magxyz;
                //float z = Random.Range(-1f, 1f) * magxyz;
                //cam.transform.localPosition += new Vector3(x, y, z);
                float rx = Random.Range(-1f, 1f) * magxyz;
                float ry = Random.Range(-1f, 1f) * magxyz;
                float rz = Random.Range(-1f, 1f) * magxyz;
                cam.transform.localEulerAngles += new Vector3(rx, ry, rz);
                //setCapStickersProperties(0.1f);
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

