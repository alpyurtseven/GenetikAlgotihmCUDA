using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.CPU;
using ILGPU.Runtime.Cuda;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace GenetikAlgCUDA
{
    class Program
    {

        static void Main(string[] args)
        {
            Genetic gn = new Genetic(100, 0.5, 10);
            Gcuda gc = new Gcuda(100, 0.5, 10);     
           
        }

      
    }
}
