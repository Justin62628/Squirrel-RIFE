using System.Collections.Generic;
using System.IO;

namespace RIFE_APP
{
    class JsonMaker
    {
        //制作JSON文件用于记录(前一帧，后一帧，exp，需要帧数)
        public void Make(string file,List<JsonItem> items)
        {
            string s = "[\r\n";
            foreach (JsonItem j in items)
            {
                s += "{" + $"\"imgs\":\"{j.imgs}\",\"exp\":\"{j.exp}\",\"need\":\"{j.need}\"" + "},\r\n";
            }
            s = s.Substring(0,s.Length-3) + "\r\n]";
            File.WriteAllText(file,s);
        }
    }
    public class JsonItem
    {
        public string imgs;
        public int exp;
        public string need;
        public JsonItem( string i, int e,string n)
        {
            exp = e;
            imgs = i;
            need = n;
        }
    }
}
