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
        string[] names = Globals.getLandmarkNames();
        Vector3[] stickers_locs = new Vector3[names.Length];
        for (int i = 0; i < names.Length; i++)
        {
            Vector3 sticker_3dloc = GameObject.Find(names[i]).transform.position;
            Vector3 sticker_2dloc = Camera.main.WorldToScreenPoint(sticker_3dloc);
            stickers_locs[i] = sticker_2dloc;
        }
        bool[] valid_stickers = { false, false, false, false, false, false, false, false, false, false };

        for (int i = 0; i < names.Length; i++)
        {
            GameObject obj = GameObject.Find(names[i]);
            var direction = (Camera.main.transform.position - obj.transform.position).normalized;
            if (Vector3.Dot(obj.transform.up, direction) >= 0)
            {
                valid_stickers[i] = true;
            }
        }
        string mystring = "";
        string s;
        for (int i = 0; i < names.Length; i++)
        {
            if (stickers_locs[i].z >= 0)
            {
                s = string.Format("{0}: {1} {2} {3}\n",
                    names[i],
                    stickers_locs[i].x.ToString("F1"),
                    stickers_locs[i].y.ToString("F1"),
                    valid_stickers[i].ToString());
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
