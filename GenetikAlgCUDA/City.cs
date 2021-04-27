using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GenetikAlgCUDA
{
    public class City
    {
        public int Number { get; set; }
        public double X { get; set; }
        public double Y { get; set; }

        public City(int number, double x, double y)
        {
            this.Number = number;
            this.X = x;
            this.Y = y;
        }
    }
}
