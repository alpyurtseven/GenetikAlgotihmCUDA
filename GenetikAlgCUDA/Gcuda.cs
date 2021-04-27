using Accord.Math;
using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace GenetikAlgCUDA
{
    class Gcuda
    {
        public int PopSize { get; set; }
        public double MutationRate { get; set; }
        public int[,] Population { get; set; }

        public List<City> Graph = new List<City>();

        private static List<City> cities1;
        int[,] citiesArr = new int[52,2];

        List<double> results = new List<double>();

        private Random rnd = new Random();
        int parentAindis = 0;
        int parentBindis = 0;
        List<int> parentsG = new List<int>();
        int[] bestGen = new int[52];
        double bestDistance = 50000;
        //public Gcuda(int popSize, double mutationRate, double fitness)
        //{
        //    Stopwatch sw = new Stopwatch();
        //    sw.Start();
        //    this.PopSize = popSize;
        //    this.MutationRate = mutationRate;

        //    CreateGraph();
        //    CreateNewPopulation(this.Graph);
        //    Selection(Population);
        //    while (results.Min() > fitness)
        //    {
        //        Selection(Population);
        //        if (rnd.NextDouble() < MutationRate)
        //        {
        //            Mutate(GetRandomGen(Population));
        //        }
        //    }
        //    sw.Stop();
        //    Console.WriteLine("----------");
        //    Console.WriteLine(results.Min());
        //    Console.WriteLine(sw.ElapsedMilliseconds);
        //    Console.WriteLine("En iyi sonuç:");
        //    for (int i = 0; i < 52; i++)
        //    {
        //        Console.Write(bestGen[i] + " ");
        //    }

        //}
        static Context context = new Context();
        Accelerator accelerator = Accelerator.Create(context, Accelerator.Accelerators[1]);
        static Stopwatch sw = new Stopwatch();
        public Gcuda(int popSize, double mutationRate, int iteration)
        {
            this.PopSize = popSize;
            this.MutationRate = mutationRate;

            CreateGraph();
            CreateNewPopulation(this.Graph);
         
          
                    while (iteration > 0)
                    {
                   
                        Selection(Population);
                  
                        if (rnd.NextDouble() < MutationRate)
                        {
                            Mutate(GetRandomGen(Population));
                        }
                        iteration--;
                Console.WriteLine(sw.ElapsedTicks+"GPU");
                     
                    }
                
            Console.WriteLine("----------");
            Console.WriteLine(results.Min());

            Console.WriteLine(sw.ElapsedMilliseconds);
            Console.WriteLine("En iyi sonuç:");
            for (int i = 0; i < 52; i++)
            {
                Console.Write(bestGen[i] + " ");
            }
        }


        private void CreateGraph()
        {
            cities1 = new List<City>();
            using (StreamReader file = new StreamReader("C:\\Users\\Alperen Yurtseven\\source\\repos\\GenetikAlgCUDA\\GenetikAlgCUDA\\berlin52.tsp"))
            {
                string ln;
                int counter = 0;
                while ((ln = file.ReadLine()) != null)
                {
                    string[] str = ln.Split(' ');

                    citiesArr[counter,0] = Convert.ToInt32(str[1].Split('.')[0]);
                    citiesArr[counter,1] = Convert.ToInt32(str[2].Split('.')[0]);
                    cities1.Add(new City(Int32.Parse(str[0]), Convert.ToDouble(str[1]), Convert.ToDouble(str[2])));
                    this.Graph.Add(new City(Int32.Parse(str[0]), Convert.ToDouble(str[1]), Convert.ToDouble(str[2])));
                    counter++;
                }
                file.Close();
            }
        }

        private void Selection(int[,] pop)
        {
            int[,] parents = new int[4, 52];
            for (int i = 0; i < 4; i++)
            {
                int random = rnd.Next(0, PopSize);
                for (int j = 0; j < 52; j++)
                {
                    parents[i, j] = pop[random, j];
                }
                parentsG.Add(random);
            }
            Tournament(parents);
        }
        public void Selection(Accelerator accelerator, int[,] pop)
        {
            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1,
                ArrayView2D<int>,
                ArrayView2D<int>,
                ArrayView<int>>(SelectionKernel);

             var popS = accelerator.Allocate<int>(100, 52);
             var parents = accelerator.Allocate<int>(4, 52);
             var randoms = accelerator.Allocate<int>(4);
         
            int[] randomsArr = { rnd.Next(0, 100), rnd.Next(0, 100), rnd.Next(0, 100), rnd.Next(0, 100) };

            foreach (var item in randomsArr)
            {
                parentsG.Add(item);
            }

            popS.CopyFrom(pop, LongIndex2.Zero, LongIndex2.Zero, popS.Extent);
            randoms.CopyFrom(randomsArr, 0, LongIndex1.Zero, randoms.Extent);

            kernel(4, popS, parents, randoms);
            accelerator.Synchronize();

            var p = parents.GetAs2DArray();
            Tournament(p);
        }
        public static void SelectionKernel(
               Index1 index,
               ArrayView2D<int> population,
               ArrayView2D<int> parents,
               ArrayView<int> randoms)
        {
            for (int j = 0; j < 52; j++)
            {
                parents[index, j] = population[randoms[index], j];
            }
        }

        private void Tournament(int[,] parents)
        {
          
            int[] gen = new int[52];
            int[] parentA = new int[52];
            int[] parentB = new int[52];
            List<double> distances = new List<double>();
            List<int> selected = new List<int>();
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 52; j++)
                {
                    gen[j] = parents[i, j];
                }
                distances.Add(CalculateDistance(accelerator,gen, cities1));
            }
            for (int i = 0; i < 4; i = i + 2)
            {
                if (distances[i].CompareTo(distances[i + 1]) > distances[i + 1].CompareTo(distances[i]))
                {
                    selected.Add(i + 1);
                }
                else if (distances[i].CompareTo(distances[i + 1]) < distances[i + 1].CompareTo(distances[i]))
                {
                    selected.Add(i);
                }
                else
                {
                    selected.Add(i);
                }
            }
            for (int i = 0; i < 52; i++)
            {
                parentA[i] = parents[selected[0], i];
                parentB[i] = parents[selected[1], i];
            }
            parentAindis = selected[0];
            parentBindis = selected[1];

            CrossOver(parentA, parentB);

        }

        private void Mutate(int[] gen)
        {
            int temp = 0;
            int index = rnd.Next(0, 52);
            int index2 = rnd.Next(0, 52);
            temp = gen[index];
            gen[index] = gen[index2];
            gen[index2] = temp;
        }

        private void CrossOver(int[] parentA, int[] parentB)
        {
            int start = rnd.Next(0, parentA.Length);
            int end = rnd.Next(start + 1, parentA.Length);
            int[] gen1 = parentA.Skip(start).Take(end).ToArray();
            List<int> gen2 = new List<int>();
            int[] child = new int[52];

            for (int i = 0; i < 52; i++)
            {
                int city = parentB[i];

                if (!gen1.Contains(city))
                {
                    gen2.Add(city);
                }
            }
            gen1.CopyTo(child, 0);
            gen2.CopyTo(child, gen1.Length);
            if (CalculateDistance(accelerator,parentA, cities1).CompareTo(CalculateDistance(accelerator, parentB, cities1)) == 1)
            {
                for (int i = 0; i < 52; i++)
                {
                    Population[parentsG[parentAindis], i] = child[i];
                }
            }
            else if (CalculateDistance(accelerator, parentA, cities1).CompareTo(CalculateDistance(accelerator, parentB, cities1)) == -1)
            {
                for (int i = 0; i < 52; i++)
                {
                    Population[parentsG[parentBindis], i] = child[i];
                }
            }
            else
            {
                for (int i = 0; i < 52; i++)
                {
                    Population[parentsG[parentAindis], i] = child[i];
                }
            }
            results.Add(Fitness(Population));
            parentsG.Clear();
            Console.WriteLine(Fitness(Population));
        }

        private List<City> Shuffle(List<City> cities)
        {

            int n = cities.Count;
            while (n > 1)
            {
                n--;
                int k = rnd.Next(n + 1);
                City value = cities[k];
                cities[k] = cities[n];
                cities[n] = value;
            }

            return cities;
        }

        public void CreateNewPopulation(List<City> cities)
        {
            int[,] population = new int[this.PopSize, 52];

            for (int i = 0; i < this.PopSize; i++)
            {
                var cities1 = Shuffle(cities);
                for (int j = 0; j < 52; j++)
                {
                    population[i, j] = cities1[j].Number;
                }
            }
            this.Population = population;
        }

        //public double CalculateDistance(int[] gen, List<City> cities)
        //{
        //    double totalDistance = 0;
        //    for (int i = 0; i < 52; i++)
        //    {
        //        if (i == 51)
        //        {
        //            totalDistance += Distance.Euclidean(new double[] { cities[gen[i]].X, cities[gen[i]].Y }, new double[] { cities[gen[0]].X, cities[gen[0]].Y });
        //        }
        //        else totalDistance += Distance.Euclidean(new double[] { cities[gen[i]].X, cities[gen[i]].Y }, new double[] { cities[gen[i + 1]].X, cities[gen[i + 1]].Y });
        //    }

        //    return totalDistance;
        //}

        public double CalculateDistance(Accelerator accelerator,int[] gen, List<City> cities)
        {

            var kernel = accelerator.LoadAutoGroupedStreamKernel<
                Index1,
                ArrayView<int>,
                ArrayView2D<int>,
                ArrayView<double>>(DistanceKernel);

            var genG = accelerator.Allocate<int>(52);
            var citiesG = accelerator.Allocate<int>(52,2);
            genG.CopyFrom(gen, 0, LongIndex1.Zero, genG.Extent);
            citiesG.CopyFrom(citiesArr, LongIndex2.Zero, LongIndex2.Zero, citiesG.Extent);
            using (var resultG = accelerator.Allocate<double>(52)) {

               
                kernel(52, genG.View, citiesG, resultG);
                sw.Restart();
                accelerator.Synchronize();
                sw.Stop();

                var p = resultG.GetAsArray();
                return p.Sum();
            }     
            
        }

        public static void DistanceKernel(Index1 index, ArrayView<int> gen, ArrayView2D<int> cities, ArrayView<double> result)
        {
            double sum = 0;
           
            if (index == 51)
            {
              sum +=Distance.Euclidean(new double[] { cities[gen[index], 0], cities[gen[index], 1] }, new double[] { cities[gen[0], 0], cities[gen[0], 1] });
            }
            else
            {
              sum += Distance.Euclidean(new double[] { cities[gen[index], 0], cities[gen[index], 1] }, new double[] { cities[gen[index + 1], 0], cities[gen[index + 1], 1] });
            }

            
            result[index] = sum;
        }


        public int[] GetRandomGen(int[,] population)
        {
            int random = rnd.Next(0, PopSize);
            int[] gen = new int[52];
            for (int i = 0; i < 52; i++)
            {
                gen[i] = population[random, i];
            }

            return gen;
        }

        public double Fitness(int[,] population)
        {

            List<double> distances = new List<double>();
            int[] gen = new int[52];
            for (int i = 0; i < PopSize; i++)
            {
                for (int j = 0; j < 52; j++)
                {
                    gen[j] = population[i, j];
                }
                double fitnessValue = CalculateDistance(accelerator, gen, cities1);
                CheckBest(gen, fitnessValue);
                distances.Add(fitnessValue);
            }
            return distances.Min();

        }
        public void CheckBest(int[] gen, double distance)
        {

            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestGen = gen;
                double bestgendistance = CalculateDistance(accelerator, bestGen, cities1);
            }
        }

    }
}
