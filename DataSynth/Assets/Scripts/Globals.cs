using UnityEngine;

public class Globals
{
	private static string[] landmarkNames = new string[] { "lefteye", "nosetip", "righteye", "leftear", "rightear", "nosebridge", "left_triangle", "middle_triangle", "right_triangle", "top" };
	public static string[] getLandmarkNames()
    {
        return landmarkNames;
    }
}
