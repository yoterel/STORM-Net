using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

//[ExecuteInEditMode]
public class textUpdater : MonoBehaviour
{
    private TextMeshProUGUI textMesh;
    // Start is called before the first frame update
    void Start()
    {
        textMesh = GetComponent<TextMeshProUGUI>();
    }

    // Update is called once per frame
    void Update()
    {
        Vector3[] stickers_locs = new Vector3[7];
        string[] names = new string[] { "AL", "NZ", "AR", "FP1", "FPZ", "FP2", "CZ" };
        for (int i = 0; i < 7; i++)
        {
            Vector3 sticker_3dloc = GameObject.Find(names[i]).transform.position;
            Vector3 sticker_2dloc = Camera.main.WorldToScreenPoint(sticker_3dloc);
            stickers_locs[i] = sticker_2dloc;
        }
        string mystring = "";
        string s;
        for (int i = 0; i < 7; i++)
        {
            if (stickers_locs[i].z >= 0)
            {
                s = string.Format("{0}: {1} {2}\n", names[i], stickers_locs[i].x.ToString("F1"), stickers_locs[i].y.ToString("F1"));
            }
            else
            {
                s = string.Format("{0}: 0 0", names[i]);
            }
            mystring = string.Concat(mystring, s);
        }
        textMesh.text = mystring;
    }
}
