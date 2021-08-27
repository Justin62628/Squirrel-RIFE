namespace ProjectFFMPEG
{
    //先前丢失的代码，使用反编译恢复

    // Token: 0x0200000D RID: 13
    public class Size
    {
        // Token: 0x06000065 RID: 101 RVA: 0x00002214 File Offset: 0x00000414
        public Size(double w, double h)
        {
            this.Width = w;
            this.Height = h;
        }

        // Token: 0x06000066 RID: 102 RVA: 0x0000666C File Offset: 0x0000486C
        public Size()
        {
            this.Width = 0.0;
            this.Height = 0.0;
        }

        // Token: 0x06000067 RID: 103 RVA: 0x0000224A File Offset: 0x0000044A
        public override string ToString()
        {
            return string.Format("({0},{1})", this.Width, this.Height);
        }

        // Token: 0x0400005B RID: 91
        public double Width = 0.0;

        // Token: 0x0400005C RID: 92
        public double Height = 0.0;
    }
}
