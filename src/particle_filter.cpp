/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "multiv_gauss.h"
#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles

  std::default_random_engine gen;
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle particle_;
    particle_.id = i;
    particle_.weight = 1.0;
    particle_.x = dist_x(gen);
    particle_.y = dist_y(gen);
    particle_.theta = dist_theta(gen);
    particles.push_back(particle_);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  double x;
  double y;
  double theta;
  std::default_random_engine gen;

  for (unsigned int i = 0; i < particles.size(); i++) {
    if (yaw_rate != 0) {
      x = particles[i].x
          + (velocity/yaw_rate)
          * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
      y = particles[i].y
          + (velocity/yaw_rate)
          * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
      theta = particles[i].theta + (yaw_rate * delta_t);
    } else {
      x = particles[i].x + ((velocity * delta_t) * cos(particles[i].theta));
      y = particles[i].y + ((velocity * delta_t) * sin(particles[i].theta));
      theta = particles[i].theta;
    }

    std::normal_distribution<double> dist_x(x, std_pos[0]);
    std::normal_distribution<double> dist_y(y, std_pos[1]);
    std::normal_distribution<double> dist_theta(theta, std_pos[2]);

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  double distance_, min_distance_;

  for (unsigned int i = 0; i < observations.size(); i++) {
    min_distance_ = std::numeric_limits<double>::infinity();

    for (unsigned int j = 0; j < predicted.size(); j++) {
      distance_ = dist(observations[i].x, observations[i].y,
                       predicted[j].x, predicted[j].y);
      if (min_distance_ > distance_) {
        min_distance_ = distance_;
        observations[i].id = predicted[j].id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (unsigned int i = 0; i < particles.size(); i++) {

    // Extract landmarks which are located within sensor_range.
    vector<LandmarkObs> predicted;
    for (unsigned int k = 0; k < map_landmarks.landmark_list.size(); k++) {
      double x = map_landmarks.landmark_list[k].x_f;
      double y = map_landmarks.landmark_list[k].y_f;
      if (dist(particles[i].x, particles[i].y, x, y) > sensor_range) {
        continue;
      }
      LandmarkObs landmark;
      landmark.id = map_landmarks.landmark_list[k].id_i;
      landmark.x = x;
      landmark.y = y;
      predicted.push_back(landmark);
    }

    // Transform x and y from the CAR's to the MAP's coordinate system.
    vector<LandmarkObs> t_observations;
    for (unsigned int j = 0; j < observations.size(); j++) {
      LandmarkObs obs = observations[j];
      LandmarkObs t_obs;
      t_obs.x = particles[i].x + (cos(particles[i].theta) * obs.x)
                               - (sin(particles[i].theta) * obs.y);
      t_obs.y = particles[i].y + (sin(particles[i].theta) * obs.x)
                               + (cos(particles[i].theta) * obs.y);
      t_observations.push_back(t_obs);
    }

    // Find a nearest neighbor for each observation.
    dataAssociation(predicted, t_observations);


    // Update particles with t_observations.
    vector<int> associations;
    vector<double> sense_x;
    vector<double> sense_y;
    for (unsigned int j = 0; j < t_observations.size(); j++) {
      LandmarkObs obs = t_observations[j];
      associations.push_back(obs.id);
      sense_x.push_back(obs.x);
      sense_y.push_back(obs.y);
    }
    setAssociations(particles[i], associations, sense_x, sense_y);

    // Calculate weight;    
    Particle particle = particles[i];
    double weight = 1.0;
    for (unsigned int j = 0; j < particles[i].associations.size(); j++) {
      weight *= multiv_prob(
          std_landmark[0], std_landmark[1],
          particles[i].sense_x[j], particles[i].sense_y[j],
          map_landmarks.landmark_list[particles[i].associations[j]-1].x_f,
          map_landmarks.landmark_list[particles[i].associations[j]-1].y_f);
    }
    particles[i].weight = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> particles_(num_particles);
  vector<double> weights(num_particles);
  std::default_random_engine gen;

  for (unsigned int i = 0; i < particles.size(); i++) {
    weights[i] = particles[i].weight;
  }
  std::discrete_distribution<int> dist_(weights.begin(), weights.end());

  for (int i = 0; i < num_particles; i++) {
    particles_[i] = particles[dist_(gen)];
  }

  particles = particles_;
}

void ParticleFilter::setAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}