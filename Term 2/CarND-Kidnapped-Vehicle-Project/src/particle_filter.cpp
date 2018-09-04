/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//  x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	if (is_initialized)
	{
		return;
	}

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);
	std::default_random_engine generator;
	num_particles = 50;
	particles.resize(num_particles);

	for (unsigned int i = 0; i < num_particles; ++i)
	{
		particles[i].id = i;
		particles[i].x = dist_x(generator);
		particles[i].y = dist_y(generator);
		particles[i].theta = dist_theta(generator);
		particles[i].weight = 1.0;
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	normal_distribution<double> dist_x(0.0, std_pos[0]);
	normal_distribution<double> dist_y(0.0, std_pos[1]);
	normal_distribution<double> dist_theta(0.0, std_pos[2]);
	std::default_random_engine generator;

	for (unsigned int i = 0; i < num_particles; ++i)
	{
		if (fabs(yaw_rate) < 0.00001)
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(generator);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(generator);
			particles[i].theta = particles[i].theta + dist_theta(generator);
		}
		else
		{
			particles[i].x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(generator);
			particles[i].y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + dist_y(generator);
			particles[i].theta = particles[i].theta + yaw_rate * delta_t + dist_theta(generator);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.
	for (unsigned int i = 0; i < observations.size(); ++i)
	{
		double min_dist = numeric_limits<double>::max();
		for (unsigned int j = 0; j < predicted.size(); ++j)
		{
			const double distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (distance < min_dist)
			{
				min_dist = distance;
				observations[i].id = predicted[j].id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	const double sensor_range2 = pow(sensor_range, 2);
	for (unsigned int i = 0; i < num_particles; ++i)
	{
		vector<LandmarkObs> landmarks;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j)
		{
			const double distance = dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particles[i].x, particles[i].y);
			if (distance <= sensor_range2)
			{
				landmarks.push_back(LandmarkObs{map_landmarks.landmark_list[j].id_i, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f});
			}
		}

		vector<LandmarkObs> mobservations(observations.size());
		for (unsigned int j = 0; j < observations.size(); ++j)
		{
			mobservations[j].id = observations[j].id;
			mobservations[j].x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
			mobservations[j].y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;
		}

		dataAssociation(landmarks, mobservations);
		particles[i].weight = 1.0; //reset weights

		for (unsigned int j = 0; j < mobservations.size(); ++j)
		{
			double dx, dy;
			for (unsigned int k = 0; k < landmarks.size(); ++k)
			{
				if (landmarks[k].id == mobservations[j].id)
				{
					dx = mobservations[j].x - landmarks[k].x;
					dy = mobservations[j].y - landmarks[k].y;
					break;
				}
			}

			double weight = (1 / (2 * M_PI * std_landmark[0] * std_landmark[1])) * exp(-(dx * dx / (2 * std_landmark[0] * std_landmark[0]) + (dy * dy / (2 * std_landmark[1] * std_landmark[1]))));
			particles[i].weight = particles[i].weight * weight;
		}
	}
}

void ParticleFilter::resample()
{
	// Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//  http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::default_random_engine generator;
	double maxWeight = numeric_limits<double>::min();
	for (unsigned int i = 0; i < num_particles; ++i)
	{
		if (particles[i].weight > maxWeight)
		{
			maxWeight = particles[i].weight;
		}
	}

	uniform_real_distribution<double> distDouble(0.0, maxWeight);
	uniform_int_distribution<int> distInt(0, num_particles - 1);
	int index = distInt(generator);
	double beta = 0.0;

	vector<Particle> resampled(num_particles);
	for (unsigned int i = 0; i < num_particles; ++i)
	{
		beta = beta + distDouble(generator) * 2.0;
		while (beta > particles[index].weight)
		{
			beta = beta - particles[index].weight;
			index = (index + 1) % num_particles;
		}

		resampled[i] = particles[index];
	}

	particles = resampled;
}

void ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
									 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
