using UnityEngine;

public class Globals
{
	private static string[] landmarkNames = new string[] { "lefteye", "nosetip", "righteye", "left_triangle", "middle_triangle", "right_triangle", "top" };
    //private static string[] landmarkNames = new string[] { "lefteye", "nosetip", "righteye", "leftear", "rightear", "nosebridge", "left_triangle", "middle_triangle", "right_triangle", "top" };
    private static float cosineThreshold = 0.3f;

    public static string[] getLandmarkNames()
    {
        return landmarkNames;
    }

    public static float getCosineThreshold()
    {
        return cosineThreshold;
    }
}
