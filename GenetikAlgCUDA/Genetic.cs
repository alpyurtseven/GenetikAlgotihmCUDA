using Accord.Math;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace GenetikAlgCUDA
{
    public class Genetic
    {
        public int PopSize { get; set; }
        public double MutationRate { get; set; }
        public int[,] Population { get; set; }

        public List<City> Graph = new List<City>();

        private static List<City> cities1;

        List<double> results = new List<double>();

        private Random rnd = new Random();
        int parentAindis = 0;
        int parentBindis = 0;
        List<int> parentsG = new List<int>();
        int[] bestGen = new int[52];
        double bestDistance = 50000;
        Stopwatch sw = new Stopwatch();
        public Genetic(int popSize,double mutationRate,double fitness)
        {
          
            sw.Start();
            this.PopSize = popSize;
            this.MutationRate = mutationRate;

            CreateGraph();
            CreateNewPopulation(this.Graph);
            Selection(Population);
            while (results.Min() > fitness)
            {
                Selection(Population);
                if (rnd.NextDouble() < MutationRate)
                {
                    Mutate(GetRandomGen(Population));
                }
            }
            sw.Stop();
            Console.WriteLine("----------");
            Console.WriteLine(results.Min());
            Console.WriteLine(sw.ElapsedMilliseconds);
            Console.WriteLine("En iyi sonuç:");
            for (int i = 0; i < 52; i++)
            {
                Console.Write(bestGen[i]+" ");
            }

        }
        public Genetic(int popSize, double mutationRate ,int iteration)
        {
            this.PopSize = popSize;
            this.MutationRate = mutationRate;

            CreateGraph(); 
            CreateNewPopulation(this.Graph);
            Stopwatch sw = new Stopwatch();
           
            while (iteration > 0)
            {
           
                Selection(Population);
                
                if (rnd.NextDouble() < MutationRate)
                {
                    Mutate(GetRandomGen(Population));
                }
                iteration--;
                Console.WriteLine(sw.ElapsedTicks +"cpu");
            }
        
            Console.WriteLine("----------");
            Console.WriteLine(results.Min());
            Console.WriteLine("En iyi sonuç:");
            for (int i = 0; i < 52; i++)
            {
                Console.Write(bestGen[i] + " ");
            }
        }

        private void CreateGraph()
        {
            cities1 = new List<City>();
            using (StreamReader file = new StreamReader("C:\\Users\\Alperen Yurtseven\\source\\repos\\GenetikAlgCUDA\\GenetikAlgCUDA\\berlin52.tsp")) //Path of the "berlin52.tsp"
            {
                string ln;
                while ((ln = file.ReadLine()) != null)
                {
                    string[] str = ln.Split(' ');
                    cities1.Add(new City(Int32.Parse(str[0]), Convert.ToDouble(str[1]), Convert.ToDouble(str[2])));
                    this.Graph.Add(new City(Int32.Parse(str[0]), Convert.ToDouble(str[1]), Convert.ToDouble(str[2])));
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
                distances.Add(CalculateDistance(gen, cities1));
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
            if (CalculateDistance(parentA, cities1).CompareTo(CalculateDistance(parentB, cities1)) == 1)
            {
                for (int i = 0; i < 52; i++)
                {
                    Population[parentsG[parentAindis], i] = child[i];
                }
            }
            else if (CalculateDistance(parentA, cities1).CompareTo(CalculateDistance(parentB, cities1)) == -1)
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

        public double CalculateDistance(int[] gen, List<City> cities)
        {
            sw.Restart();
            double totalDistance = 0;
            for (int i = 0; i < 52; i++)
            {
                if (i == 51)
                {
                    totalDistance += Distance.Euclidean(new double[] { cities[gen[i]].X, cities[gen[i]].Y }, new double[] { cities[gen[0]].X, cities[gen[0]].Y });
                }
                else totalDistance += Distance.Euclidean(new double[] { cities[gen[i]].X, cities[gen[i]].Y }, new double[] { cities[gen[i + 1]].X, cities[gen[i + 1]].Y });
            }

            sw.Stop();
            Console.WriteLine(sw.ElapsedTicks+"Distance");
            return totalDistance;
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
                double fitnessValue = CalculateDistance(gen, cities1);
                CheckBest(gen, fitnessValue);
                distances.Add(fitnessValue); 
            }
            return distances.Min();
          
        }
        public void CheckBest(int[] gen,double distance)
        {
           
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestGen = gen;
                double bestgendistance = CalculateDistance(bestGen, cities1);
            }
        }
        

        
    }
}