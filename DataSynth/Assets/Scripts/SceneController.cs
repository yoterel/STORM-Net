using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SceneController : MonoBehaviour
{
    public ImageSynthesis synth;
    //cmd line arguments
    private int numberOfIterations;
    private bool saveImage;
    private bool saveData;
    private bool shiftCamera;
    private bool rotateCamera;
    private string inputFile;
    private string outputFolder;
    //private parameters
    private Camera cam;
    private GameObject camHolder;
    private string[] stickerNames;
    private Dictionary<string, Vector3> stickerDictionary;
    private Vector3 camHolderInitialPosition;
    private Vector3 targetCamHolderPosition;
    private Vector3 camHolderInitialRotation;
    private Vector3 faceInitialPosition;
    private Vector3 faceInitialRotation;
    private Vector3 maskInitialPosition;
    private Vector3[] initial_sticker_positions;
    private int frameCounter;
    private int iterationCount;
    private GameObject face;
    private GameObject mask;
    private RotationPaths stage;
    private float speed; // determines how many frames will be captured during 1 iteration
    private int finalFaceAngle; // final angle of face, determined hueristically
    private float shakeyCamMagnitude; //magnitude of shakey cam effect
    private Quaternion startOrientation;
    private int angleAmount;
    private int img_width;
    private int img_height;
    private bool doingRotation;
    private bool iterationComplete;
    private int magicNumber;

    void Awake()
    {
        stickerNames = new string[] { "AL", "NZ", "AR", "FP1", "FPZ", "FP2", "CZ" };
        Dictionary<string, Vector3> stickerDictionary = new Dictionary<string, Vector3>();
        foreach (string sticker in stickerNames)
        {
            stickerDictionary.Add(sticker, Vector3.zero);
        }
        camHolderInitialPosition = new Vector3(0f, 25f, 0);
        camHolderInitialRotation = new Vector3(90f, 0f, 0f);
        faceInitialPosition = new Vector3(0f, 0f, 0f);
        faceInitialRotation = new Vector3(0f, 0f, 0f);
        maskInitialPosition = new Vector3(0f, 0f, 0f);
        initial_sticker_positions = new Vector3[7];
        frameCounter = 0;
        iterationCount = 0;
        stage = RotationPaths.front_to_up;
        speed = 400;
        finalFaceAngle = 80;
        shakeyCamMagnitude = 3f;
        doingRotation = false;
        iterationComplete = false;
        img_width = 960;
        img_height = 540;
        magicNumber = 5;
        face = GameObject.Find("face");
        mask = GameObject.Find("mask");
        camHolder = GameObject.Find("CameraHolder");
        cam = Camera.main;
    }
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
    private void parseCommandLine()
    {
        System.Console.WriteLine("Parsing command line options");
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
        iterationsString = GetArg("-input_file");
        if (string.IsNullOrEmpty(iterationsString))
        {
            inputFile = "default";
        }
        else
        {
            inputFile = iterationsString;
        }
        iterationsString = GetArg("-output_folder");
        if (string.IsNullOrEmpty(iterationsString))
        {
            outputFolder = "captures";
        }
        else
        {
            outputFolder = iterationsString;
        }
        System.Console.WriteLine("Running {0} iterations", numberOfIterations);
        System.Console.WriteLine("Saving image to disk == {0}", saveImage);
        System.Console.WriteLine("Saving data to disk == {0}", saveData);
        System.Console.WriteLine("Camera shift enabled == {0}", shiftCamera);
        System.Console.WriteLine("Camera rotation enabled == {0}", rotateCamera);
        System.Console.WriteLine("Input file: " + inputFile);
        System.Console.WriteLine("Output folder: " + outputFolder);
    }

    private Vector3[] readStickerLocations()
    {
        Vector3[] positions = new Vector3[7];
        bool[] filled = new bool[7] { false, false, false, false, false, false, false };
        Vector3[] defaultPositions = new Vector3[] {
                new Vector3 {x = 2.44f, y = 5.69f, z = 0f },
                new Vector3 {x = 0.13f, y = 7.34f, z = -1.21f },
                new Vector3 {x = -2.2f, y = 6.12f, z = 0.02f },
                new Vector3 {x = 3.5f, y = 9.75f, z = 4.66f },
                new Vector3 {x = -0.36f, y = 9.33f, z = 6.68f },
                new Vector3 {x = -3.77f, y = 9.74f, z = 4.87f },
                new Vector3 {x = -0.53f, y = 0f, z = 10.98f }};
        if (inputFile == "default")
        {
            return defaultPositions;
        }
        else
        {
            string[] alias = { "LeftEye", "NoseTip", "RightEye", "Fp1", "Fpz", "Fp2", "Cz" };
            string fileData = System.IO.File.ReadAllText(inputFile);
            string[] lines = fileData.Split("\n"[0]);
            for (int i = 0; i < lines.Length; i++)
            {
                string[] lineData = (lines[i].Trim()).Split(" "[0]);
                if (lineData[0] != "")
                {
                    string name = lineData[0];
                    int index = System.Array.IndexOf(alias, name);
                    if (index != -1)
                    {
                        float x, y, z;
                        float.TryParse(lineData[1], out x);
                        float.TryParse(lineData[2], out y);
                        float.TryParse(lineData[3], out z);
                        positions[index] = new Vector3 { x = x, y = y, z = z };
                        filled[index] = true;
                    }
                }
            }
            foreach (bool entry in filled)
            {
                System.Console.WriteLine("Parsing entry {0}", entry);
                if (!entry)
                {
                    System.Console.WriteLine("Error finding sticker with alias: " + name + ". Falling back to default.");
                    return defaultPositions;
                }
            }
            /* user is instructed to input data where the x axis positive direction is from left eye to right eye (right handed system),
               yet simulation uses oposite direction intrinsically (left handed system) */
            for (int i = 0; i < positions.Length; i++)
            {
                positions[i].x *= -1;
            }
        }
        return positions;

    }

    private bool verifyStickerPositions(Vector3[] positions)
    {
        // just some sanity checks on input
        if (positions[0].x < positions[2].x)
        {
            System.Console.WriteLine("Sanity check failed. Left-Eye sticker x smaller than Right-Eye sticker.");
            return false;
        }
        if (positions[6].y != 0f)
        {
            System.Console.WriteLine("Sanity check failed. CZ is not centered in y axis.");
            return false;
        }
        if (positions[1].y < positions[0].y || positions[1].y < positions[2].y)
        {
            System.Console.WriteLine("Sanity check failed. Nose sticker is inside of skull.");
            return false;
        }
        if (positions[6].z < positions[4].z)
        {
            System.Console.WriteLine("Sanity check failed. CZ is not the highest sticker in z direction.");
            return false;
        }
        return true;
    }

    private Vector3[] centerStickerPostions(Vector3[] positions)
    {
        Vector3[] centeredPositions = positions;
        //todo: estimate better center of brain (nose bridge, Inion, right ear and left ear)
        //todo: add scaling and rotating
        float my_x = (positions[0].x + positions[2].x) / 2;
        float my_y = positions[6].y;
        float my_z = (positions[0].z + positions[2].z) / 2;
        // calc "center of brain" as pivoting point
        Vector3 pivot_point = new Vector3(my_x, my_y, my_z);
        for (int i = 0; i < positions.Length; i++)
        {
            centeredPositions[i] -= pivot_point;
        }
        return centeredPositions;
    }
    void Start()
    {
        parseCommandLine();
        Vector3[] inputStickerPositions = readStickerLocations();
        Vector3[] stickerPositions = centerStickerPostions(inputStickerPositions);
        if (!verifyStickerPositions(stickerPositions))
        {
            System.Console.WriteLine("Quitting application. Reason: sanity check failed.");
            Application.Quit();
        }
        for (int i = 0; i < stickerNames.Length; i++)
        {
            GameObject sticker = GameObject.Find(stickerNames[i]);
            sticker.transform.localPosition = stickerPositions[i];
            initial_sticker_positions[i] = sticker.transform.localPosition;
        }
        //set occlusion object position using Fp2
        GameObject occlusion = GameObject.Find("Occlusion");
        Vector3 occlusion_loc = new Vector3 { x = stickerPositions[5].x - 2f, y = stickerPositions[5].y, z = stickerPositions[5].z -1.5f};
        occlusion.transform.localPosition = occlusion_loc;

        startOrientation = face.transform.rotation;
        face.transform.position = faceInitialPosition;
        face.transform.eulerAngles = faceInitialRotation;
        setFaceStickersProperties();
        setMaskProperties();
        setCameraProperties();
    }

    void setFaceStickersProperties()
    {
        //set face stickers to initial position
        //next loop assumes face stickers are the first stickers in initial_sticker_positions
        for (int i = 0; i < 3; i++)
        {
            GameObject sticker = GameObject.Find(stickerNames[i]);
            sticker.transform.position = initial_sticker_positions[i];
        }
        //set eye distance randomly but keep them in same plane as user data
        GameObject AR = GameObject.Find("AR");
        GameObject AL = GameObject.Find("AL");
        Vector3 diff = AR.transform.position - AL.transform.position;
        Vector3 add = AR.transform.position + AL.transform.position;
        Vector3 direction = diff.normalized;
        Vector3 middle_point = add / 2;
        float eye_dist = Random.Range(4f, 5.5f); //hard coded pupilary distance range
        AR.transform.position = middle_point + (direction * eye_dist / 2);
        AL.transform.position = middle_point - (direction * eye_dist / 2);
        //set nose exactly between the 2 eyes in x axis
        GameObject NZ = GameObject.Find("NZ");
        Vector3 temp = NZ.transform.position;
        temp.x = middle_point.x;
        NZ.transform.position = temp;
        //set additional random nose parameters ("drop" & "depth"), this goes ontop user data
        float nose_drop = Random.Range(0f, 1f);
        float nose_depth = Random.Range(-0.5f, 0.5f);
        NZ.transform.position += NZ.transform.TransformDirection(Vector3.back) * nose_drop;
        NZ.transform.position += NZ.transform.TransformDirection(Vector3.up) * nose_depth;
        //sanity check: if nose passes eye depth, clip it to eye depth.
        if (NZ.transform.position.y < AR.transform.position.y)
        {
            temp = NZ.transform.position;
            temp.y = AR.transform.position.y;
            NZ.transform.position = temp;
        }
    }
    void setMaskProperties()
    {
        //initialize mask location (rotation is determined randomly)
        mask.transform.position = maskInitialPosition;
        //set mask scale (not in use)
        //float randxscale = Random.Range(1f, 1.5f);
        //float randyscale = Random.Range(0.8f, 1f);
        //float randzscale = Random.Range(1f, 1.2f);
        //mask.transform.localScale = new Vector3(randxscale, randyscale, randzscale);
        //set random mask rotation (ranges were hueristicly defined by observing real data)
        float randx = Random.Range(-10f, 10f);
        float randy = Random.Range(-15f, 15f);
        float randz = Random.Range(-20f, 20f);
        mask.transform.localRotation = Quaternion.Euler(randx, randy, randz); //note: transforms mask relative to face
        //todo: what is the pivot?
    }
    void setCameraProperties()
    {
        //initialize camera&holder position & rotation
        camHolder.transform.position = camHolderInitialPosition;
        camHolder.transform.eulerAngles = camHolderInitialRotation;
        cam.transform.localPosition = Vector3.zero;
        cam.transform.localEulerAngles = Vector3.zero;
        float randx, randy, randz;
        //set camera random properties
        if (shiftCamera)
        {
            randy = Random.Range(25f, 47f);  //initial camera y position
            //float rangex = map_range(20f, 35f, 2f, 9f, randy);
            //randx = Random.Range(-rangex, rangex);
            //float rangez = map_range(20f, 35f, 2f, 10f, randy);
            //randz = Random.Range(-rangez, rangez);
            camHolder.transform.position = new Vector3(0, randy, 0);
            float rand = Random.Range(25, 47f);
            targetCamHolderPosition = camHolder.transform.position;  //the camera perfoms a linear movement in y axis between 2 random locations
            targetCamHolderPosition.y = rand;
        }
        if (rotateCamera)
        {
            //rotate camera randomly to better simulate real camera movement
            randx = Random.Range(-5f, 5f);
            randy = Random.Range(-5f, -5f);
            randz = Random.Range(-5f, 5f);
            cam.transform.localEulerAngles = new Vector3(randx, randy, randz);
        }
    }

    void FixedUpdate()
    {
        if (iterationCount < numberOfIterations)
        {
            if (!iterationComplete)
            {
                //during iteration
                facePath();
                camPath();
            }
            else
            {
                System.Console.WriteLine("Finished iteration: {0}", iterationCount);
                //end of iteration
                iterationCount++;
                frameCounter = 0;
                iterationComplete = false;
                setFaceStickersProperties();
                setMaskProperties();
                setCameraProperties();
                stage = RotationPaths.front_to_up;
            }
        }
        else
        {
            System.Console.WriteLine("Done!");
#if UNITY_EDITOR
            UnityEditor.EditorApplication.isPlaying = false;
#else
            Application.Quit();
#endif
        }
    }
    void camPath()
    {
        //move camera in linear path on y axis
        Vector3 dirNormalized = (targetCamHolderPosition - camHolder.transform.position).normalized;
        camHolder.transform.position = camHolder.transform.position + dirNormalized * Time.deltaTime * magicNumber;
    }
    void facePath()
    {
        //rotate face from front to upward position
        switch (stage)
        {
            case RotationPaths.front_to_up:
                {
                    if (doingRotation == false)
                    {
                        startOrientation = face.transform.rotation;
                        angleAmount = finalFaceAngle;
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
                        angleAmount = finalFaceAngle;
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
                synth.Save(filename, img_width, img_height, outputFolder, 1, saveImage, saveData);
                amount += Time.fixedDeltaTime * speed;
                face.transform.rotation = startOrientation * Quaternion.AngleAxis(amount, axis);
                //float x = Random.Range(-1f, 1f) * magxyz;
                //float y = Random.Range(-1f, 1f) * magxyz;
                //float z = Random.Range(-1f, 1f) * magxyz;
                //cam.transform.localPosition += new Vector3(x, y, z);
                float rx = Random.Range(-1f, 1f) * shakeyCamMagnitude;
                float ry = Random.Range(-1f, 1f) * shakeyCamMagnitude;
                float rz = Random.Range(-1f, 1f) * shakeyCamMagnitude;
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
    /*
    float map_range(float s_range_low, float s_range_high, float d_range_low, float d_range_high, float value)
    {
        float m = (d_range_high - d_range_low) / (s_range_high - s_range_low); // slope
        float b = d_range_low - (m * s_range_low);
        float my_range = m * value + b;
        return my_range;
    }
    */
    //void setCapStickersProperties(float mag)
    //{
    //    //string[] names = new string[] {"FP1", "FPZ", "FP2", "CZ"};
    //    //for (int i = 0; i < names.Length; i++)
    //    //{
    //    //    GameObject sticker = GameObject.Find(names[i]);
    //    //    sticker.transform.localPosition = Vector3.zero;
    //    //    Vector2 my_random_vector = Random.insideUnitCircle * mag;
    //    //    sticker.transform.position += sticker.transform.TransformDirection(Vector3.right) * my_random_vector.x;
    //    //    sticker.transform.position += sticker.transform.TransformDirection(Vector3.forward) * my_random_vector.y;
    //    //}
    //}
}

