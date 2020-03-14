using UnityEngine;
using UnityEngine.Rendering;
using System.Collections;
using System.IO;
// @TODO:
// . support custom color wheels in optical flow via lookup textures
// . support custom depth encoding
// . support multiple overlay cameras
// . tests
// . better example scene(s)

// @KNOWN ISSUES
// . Motion Vectors can produce incorrect results in Unity 5.5.f3 when
//      1) during the first rendering frame
//      2) rendering several cameras with different aspect ratios - vectors do stretch to the sides of the screen

[RequireComponent (typeof(Camera))]
[ExecuteInEditMode]
public class ImageSynthesis : MonoBehaviour {
    public Shader uberReplacementShader;

    struct CapturePass
    {
        // configuration
        public string name;
        public bool supportsAntialiasing;
        public bool needsRescale;
        public CapturePass(string name_) { name = name_; supportsAntialiasing = true; needsRescale = false; camera = null; }

        // impl
        public Camera camera;
    };
    // pass configuration
    private CapturePass[] capturePasses = new CapturePass[] {
		new CapturePass() { name = "_img" },
		new CapturePass() { name = "_layer", supportsAntialiasing = false }
	};
    void Start()
	{
        // default fallbacks, if shaders are unspecified
        if (!uberReplacementShader)
			uberReplacementShader = Shader.Find("Hidden/UberReplacement");

		// use real camera to capture final image
		capturePasses[0].camera = GetComponent<Camera>();
		for (int q = 1; q < capturePasses.Length; q++)
			capturePasses[q].camera = CreateHiddenCamera (capturePasses[q].name);

		OnCameraChange();
		OnSceneChange();
	}

	void LateUpdate()
	{
		#if UNITY_EDITOR
		if (DetectPotentialSceneChangeInEditor())
			OnSceneChange();
		#endif // UNITY_EDITOR

		// @TODO: detect if camera properties actually changed
		OnCameraChange();
	}
	
	private Camera CreateHiddenCamera(string name)
	{
		var go = new GameObject (name, typeof (Camera));
		go.hideFlags = HideFlags.HideAndDontSave;
		go.transform.parent = transform;

		var newCamera = go.GetComponent<Camera>();
		return newCamera;
	}


	static private void SetupCameraWithReplacementShader(Camera cam, Shader shader, ReplacelementModes mode)
	{
		SetupCameraWithReplacementShader(cam, shader, mode, Color.black);
	}

	static private void SetupCameraWithReplacementShader(Camera cam, Shader shader, ReplacelementModes mode, Color clearColor)
	{
		var cb = new CommandBuffer();
		cb.SetGlobalFloat("_OutputMode", (int)mode); // @TODO: CommandBuffer is missing SetGlobalInt() method
		cam.AddCommandBuffer(CameraEvent.BeforeForwardOpaque, cb);
		cam.AddCommandBuffer(CameraEvent.BeforeFinalPass, cb);
		cam.SetReplacementShader(shader, "");
		cam.backgroundColor = clearColor;
		cam.clearFlags = CameraClearFlags.SolidColor;
	}

	static private void SetupCameraWithPostShader(Camera cam, Material material, DepthTextureMode depthTextureMode = DepthTextureMode.None)
	{
		var cb = new CommandBuffer();
		cb.Blit(null, BuiltinRenderTextureType.CurrentActive, material);
		cam.AddCommandBuffer(CameraEvent.AfterEverything, cb);
		cam.depthTextureMode = depthTextureMode;
	}

	enum ReplacelementModes {
		ObjectId 			= 0,
		CatergoryId			= 1,
		DepthCompressed		= 2,
		DepthMultichannel	= 3,
		Normals				= 4
	};

	public void OnCameraChange()
	{
		int targetDisplay = 1;
		var mainCamera = GetComponent<Camera>();
		foreach (var pass in capturePasses)
		{
			if (pass.camera == mainCamera)
				continue;

			// cleanup capturing camera
			pass.camera.RemoveAllCommandBuffers();

			// copy all "main" camera parameters into capturing camera
			pass.camera.CopyFrom(mainCamera);

			// set targetDisplay here since it gets overriden by CopyFrom()
			pass.camera.targetDisplay = targetDisplay++;
		}

		// setup command buffers and replacement shaders
		SetupCameraWithReplacementShader(capturePasses[1].camera, uberReplacementShader, ReplacelementModes.CatergoryId);
	}


	public void OnSceneChange()
	{
		var renderers = Object.FindObjectsOfType<Renderer>();
		var mpb = new MaterialPropertyBlock();
		foreach (var r in renderers)
		{
			var id = r.gameObject.GetInstanceID();
			var layer = r.gameObject.layer;
			var tag = r.gameObject.tag;

			mpb.SetColor("_ObjectColor", ColorEncoding.EncodeIDAsColor(id));
			mpb.SetColor("_CategoryColor", ColorEncoding.EncodeLayerAsColor(layer-8));
			r.SetPropertyBlock(mpb);
		}
	}

	public void Save(string filename, int width = -1, int height = -1, string path = "", int specificPass = -1, bool saveImage = false, bool saveData = true)
	{
		if (width <= 0 || height <= 0)
		{
			width = Screen.width;
			height = Screen.height;
		}

		var filenameExtension = System.IO.Path.GetExtension(filename);
		if (filenameExtension == "")
			filenameExtension = ".png";
		var filenameWithoutExtension = Path.GetFileNameWithoutExtension(filename);

		var pathWithoutExtension = Path.Combine(path, filenameWithoutExtension);

		// execute as coroutine to wait for the EndOfFrame before starting capture
		StartCoroutine(
			WaitForEndOfFrameAndSave(pathWithoutExtension, filenameExtension, width, height, specificPass, saveImage, saveData));
	}

	private IEnumerator WaitForEndOfFrameAndSave(string filenameWithoutExtension, string filenameExtension, int width, int height, int specificPass, bool saveImage, bool saveData)
	{
		yield return new WaitForEndOfFrame();
		Save(filenameWithoutExtension, filenameExtension, width, height, specificPass, saveImage, saveData);
	}

	private void Save(string filenameWithoutExtension, string filenameExtension, int width, int height, int specificPass, bool saveImage, bool saveData)
	{
        if (specificPass == -1)
        {
		    foreach (var pass in capturePasses)
			    Save(pass.camera, filenameWithoutExtension + pass.name + filenameExtension, width, height, pass.supportsAntialiasing, pass.needsRescale, saveImage, saveData);
        }
        else
        {
            //var pass = capturePasses[0];
            //Save(pass.camera, filenameWithoutExtension + pass.name + filenameExtension, width, height, pass.supportsAntialiasing, pass.needsRescale);
            var pass = capturePasses[specificPass];
            Save(pass.camera, filenameWithoutExtension + pass.name + filenameExtension, width, height, pass.supportsAntialiasing, pass.needsRescale, saveImage, saveData);
        }
	}

	private void Save(Camera cam, string filename, int width, int height, bool supportsAntialiasing, bool needsRescale, bool saveImage = true, bool saveData = true)
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
        if (saveData)
        {
            bool[] valid_stickers = { false, false, false, false, false, false, false };
            Color[] pix = tex.GetPixels();
            for (int i=0; i<7; i++)
            {
                Color color = ColorEncoding.EncodeLayerAsColor(i);
                bool exists = doesColorExists(pix, color);
                valid_stickers[i] = exists;
            }
            saveJson(filename, cam, valid_stickers, width, height);
        }
        //encode texture into PNG
        if (saveImage)
        {
            var bytes = tex.EncodeToPNG();
            File.WriteAllBytes(filename, bytes);
        }
        // restore state and cleanup
        cam.targetTexture = prevCameraRT;
		RenderTexture.active = prevActiveRT;

		Object.Destroy(tex);
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
        Vector3[] stickers_locs = new Vector3[7];
        string[] names = new string[] { "AL", "NZ", "AR", "FP1", "FPZ", "FP2", "CZ"};
        for (int i = 0; i < 7; i++)
        {
            Vector3 sticker_3dloc = GameObject.Find(names[i]).transform.position;
            Vector3 sticker_2dloc = cam.WorldToScreenPoint(sticker_3dloc);
            if ((sticker_2dloc.x > width) || (sticker_2dloc.x < 0) || (sticker_2dloc.y > height) || (sticker_2dloc.y < 0))
                valid_stickers[i] = false; //center of object is out of screen
            stickers_locs[i] = sticker_2dloc;
        }
        Vector3 cap_rot = GameObject.Find("mask").transform.localEulerAngles;
        Vector3 scale = GameObject.Find("mask").transform.localScale;
        SaveObject obj = new SaveObject { valid_stickers = valid_stickers, stickers_locs = stickers_locs , cap_rot = cap_rot, scalex = scale.x, scaley = scale.y, scalez = scale.z};
        string json = JsonUtility.ToJson(obj);
        //Debug.Log(json);
        var filenameWithoutExtension = Path.GetFileNameWithoutExtension(filename);
        filenameWithoutExtension = filenameWithoutExtension.Substring(0, 12);
        File.AppendAllText("captures/" + filenameWithoutExtension + ".json",
                   json + System.Environment.NewLine);
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
#if UNITY_EDITOR
    private GameObject lastSelectedGO;
	private int lastSelectedGOLayer = -1;
	private string lastSelectedGOTag = "unknown";
	private bool DetectPotentialSceneChangeInEditor()
	{
		bool change = false;
		// there is no callback in Unity Editor to automatically detect changes in scene objects
		// as a workaround lets track selected objects and check, if properties that are 
		// interesting for us (layer or tag) did not change since the last frame
		if (UnityEditor.Selection.transforms.Length > 1)
		{
			// multiple objects are selected, all bets are off!
			// we have to assume these objects are being edited
			change = true;
			lastSelectedGO = null;
		}
		else if (UnityEditor.Selection.activeGameObject)
		{
			var go = UnityEditor.Selection.activeGameObject;
			// check if layer or tag of a selected object have changed since the last frame
			var potentialChangeHappened = lastSelectedGOLayer != go.layer || lastSelectedGOTag != go.tag;
			if (go == lastSelectedGO && potentialChangeHappened)
				change = true;
            
			lastSelectedGO = go;
			lastSelectedGOLayer = go.layer;
			lastSelectedGOTag = go.tag;
		}

		return change;
	}
#endif // UNITY_EDITOR
}
