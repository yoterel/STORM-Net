using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

//[ExecuteInEditMode]
public class Painter : MonoBehaviour
{
    public List<GameObject> landmarks;
    public bool initialize = false;
    public bool save = false;
    private Camera myHiddenCamera;
    // Start is called before the first frame update
    public void Initialize()
    {
        System.Console.WriteLine("Painter initialized.");
        myHiddenCamera = CreateHiddenCamera("myHiddenCamera");
    }
    // Update is called once per frame
    void Update()
    {
        if (initialize)
        {
            Initialize();
            initialize = false;
        }
        if (save)
        {
            int img_width = 960;
            int img_height = 540;
            string outputFolder = "captures";
            bool saveImage = true;
            bool saveData = true;
            Save(0, 0, img_width, img_height, outputFolder, saveImage, saveData);
            save = false;
        }
        for (int i = 0; i < landmarks.Count; i++)
        {
            var curColor = landmarks[i].GetComponent<Renderer>().sharedMaterial.color;
            var newColor = ColorEncoding.EncodeLayerAsColor(i);
            if (curColor != newColor)
            {
                var curMaterial = landmarks[i].GetComponent<Renderer>().sharedMaterial;
                var newMaterial = new Material(curMaterial);
                newMaterial.color = newColor;
                landmarks[i].GetComponent<Renderer>().sharedMaterial = newMaterial;
            }
        }
    }
    void LateUpdate()
    {
        OnCameraChange();
    }
    public void OnCameraChange()
    {
        int targetDisplay = 1;
        var mainCamera = GetComponent<Camera>();
        myHiddenCamera.RemoveAllCommandBuffers();
        myHiddenCamera.CopyFrom(mainCamera);
        myHiddenCamera.targetDisplay = targetDisplay;
    }
    private Camera CreateHiddenCamera(string name)
    {
        var go = new GameObject(name, typeof(Camera));
        go.hideFlags = HideFlags.HideAndDontSave;
        go.transform.parent = transform;

        var newCamera = go.GetComponent<Camera>();
        return newCamera;
    }
    public void Save(int iterationCount, int frameCounter, int width = -1, int height = -1, string path = "", bool saveImage = false, bool saveData = true)
    {
        if (width <= 0 || height <= 0)
        {
            width = Screen.width;
            height = Screen.height;
        }
        // execute as coroutine to wait for the EndOfFrame before starting capture
        StartCoroutine(
            WaitForEndOfFrameAndSave(iterationCount, frameCounter, path, width, height, saveImage, saveData));
    }

    private IEnumerator WaitForEndOfFrameAndSave(int iterationCount, int frameCounter, string path, int width, int height, bool saveImage, bool saveData)
    {
        yield return new WaitForEndOfFrame();
        Save(myHiddenCamera, iterationCount, frameCounter, path, width, height, false, false, saveImage, saveData);
    }

    private void Save(Camera cam, int iterationCount, int frameCounter, string path, int width, int height, bool supportsAntialiasing, bool needsRescale, bool saveImage = true, bool saveData = true)
    {
        var mainCamera = GetComponent<Camera>();
        var depth = 24;
        var format = RenderTextureFormat.Default;
        var readWrite = RenderTextureReadWrite.Default;
        var antiAliasing = (supportsAntialiasing) ? Mathf.Max(1, QualitySettings.antiAliasing) : 1;

        var finalRT =
            RenderTexture.GetTemporary(width, height, depth, format, readWrite, antiAliasing);
        var renderRT = (!needsRescale) ? finalRT :
            RenderTexture.GetTemporary(mainCamera.pixelWidth, mainCamera.pixelHeight, depth, format, readWrite, antiAliasing);
        var tex = new Texture2D(width, height, TextureFormat.ARGB32, false, false);
        var prevActiveRT = RenderTexture.active;
        var prevCameraRT = cam.targetTexture;

        renderRT.filterMode = FilterMode.Point;
        tex.filterMode = FilterMode.Point;
        // render to offscreen texture (readonly from CPU side)
        RenderTexture.active = renderRT;
        cam.targetTexture = renderRT;

        cam.Render();

        if (needsRescale)
        {
            // blit to rescale (see issue with Motion Vectors in @KNOWN ISSUES)
            RenderTexture.active = finalRT;
            Graphics.Blit(renderRT, finalRT);
            RenderTexture.ReleaseTemporary(renderRT);
        }

        // read offsreen texture contents into the CPU readable texture
        tex.ReadPixels(new Rect(0, 0, tex.width, tex.height), 0, 0);
        tex.Apply();
        // create filename
        string jsonFilename = $"image_{iterationCount.ToString().PadLeft(6, '0')}.json";
        string imgFilename = $"image_{iterationCount.ToString().PadLeft(6, '0')}_{frameCounter.ToString().PadLeft(3, '0')}.png";
        if (saveData)
        {
            bool[] valid_stickers = { false, false, false, false, false, false, false, false, false, false };
            
            for (int i = 0; i < landmarks.Count; i++)
            {
                var direction = (Camera.main.transform.position - landmarks[i].transform.position).normalized;
                if (Vector3.Dot(landmarks[i].transform.up, direction) >= Globals.getCosineThreshold())
                {
                    valid_stickers[i] = true;
                }
            }
            /*Color[] pix = tex.GetPixels();
            for (int i = 0; i < valid_stickers.Length; i++)
            {
                Color color = ColorEncoding.EncodeLayerAsColor(i);
                bool exists = doesColorExists(pix, color);
                valid_stickers[i] = exists;
            }*/
            saveJson(Path.Combine(path, jsonFilename), cam, valid_stickers, width, height);
        }
        //encode texture into PNG
        if (saveImage)
        {
            var bytes = tex.EncodeToPNG();
            File.WriteAllBytes(Path.Combine(path, imgFilename), bytes);
        }
        // restore state and cleanup
        cam.targetTexture = prevCameraRT;
        RenderTexture.active = prevActiveRT;
#if UNITY_EDITOR
        Object.DestroyImmediate(tex);
#else
        Object.Destroy(tex);
#endif
        RenderTexture.ReleaseTemporary(finalRT);
    }
    private bool doesColorExists(Color[] pix, Color color)
    {
        bool retVal = false;
        for (int i = 0; i < pix.Length; i++)
        {
            if ((int)(pix[i].r * 10) == (int)(color.r * 10)
                && (int)(pix[i].g * 10) == (int)(color.g * 10)
                && (int)(pix[i].b * 10) == (int)(color.b * 10))
                retVal = true;
        }
        return retVal;
    }
    void saveJson(string filename, Camera cam, bool[] valid_stickers, int width, int height)
    {
        string[] names = Globals.getLandmarkNames();
        Vector3[] stickers_locs = new Vector3[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            Vector3 sticker_3dloc = GameObject.Find(names[i]).transform.position;
            Vector3 sticker_2dloc = cam.WorldToScreenPoint(sticker_3dloc);
            if ((sticker_2dloc.x > width) || (sticker_2dloc.x < 0) || (sticker_2dloc.y > height) || (sticker_2dloc.y < 0))
                valid_stickers[i] = false; //center of object is out of screen
            stickers_locs[i] = sticker_2dloc;
        }
        Vector3 cap_rot = GameObject.Find("mask").transform.localEulerAngles;
        Vector3 scale = GameObject.Find("face").transform.localScale;
        SaveObject obj = new SaveObject { valid_stickers = valid_stickers, stickers_locs = stickers_locs, cap_rot = cap_rot, scalex = scale.x, scaley = scale.y, scalez = scale.z };
        string json = JsonUtility.ToJson(obj);
        //Debug.Log(json);
        var filenameWithoutExtension = Path.GetFileNameWithoutExtension(filename);
        //filenameWithoutExtension = filenameWithoutExtension.Substring(0, 12);
        var parent = System.IO.Directory.GetParent(filename).FullName;
        var full_path = Path.Combine(parent, filenameWithoutExtension + ".json");
        File.AppendAllText(full_path, json + System.Environment.NewLine);
        //File.WriteAllText("captures/" + filenameWithoutExtension + ".json", json);
    }
    private class SaveObject
    {
        public bool[] valid_stickers;
        public Vector3[] stickers_locs;
        public Vector3 cap_rot;
        public float scalex;
        public float scaley;
        public float scalez;
    }



}
