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

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  default_random_engine gen;
  // Normal distribution for x, y and theta
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  weights.resize(num_particles);
  particles.resize(num_particles);
  for(int i = 0; i < num_particles; ++i){
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1.0;
    weights[i] = 1.0;
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
  default_random_engine gen;
  
  double x_f;
  double y_f;
  double theta_f;
  double dt = delta_t;
  for(int i = 0; i< particles.size(); ++i){
    double x_0 = particles[i].x;
    double y_0 = particles[i].y;
    double theta_0 = particles[i].theta;
    //Check if the yawrate is close to zero
    if(fabs(yaw_rate)<0.001){
      x_f = x_0 + velocity*dt*cos(theta_0);
      y_f = y_0 + velocity*dt*sin(theta_0);
      theta_f = theta_0;
    }else{
      x_f = x_0 + (velocity/yaw_rate)*(sin(theta_0+yaw_rate*dt)-sin(theta_0));
      y_f = y_0 + (velocity/yaw_rate)*(cos(theta_0)-cos(theta_0+yaw_rate*dt));
      theta_f = theta_0 + yaw_rate*dt;
    }
    normal_distribution<double> dist_x(x_f, std_pos[0]);
  	normal_distribution<double> dist_y(y_f, std_pos[1]);
  	normal_distribution<double> dist_theta(theta_f, std_pos[2]);
    
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
  for (unsigned int i = 0; i < observations.size(); ++i){
    double near_nei = std::numeric_limits<float>::max();// Initialize the nearest distance by assign a large number to near_nei
    for (unsigned int j = 0; j < predicted.size();++j){
      double distance = dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
      if(near_nei > distance){
        near_nei = distance;
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
  for(unsigned int i = 0; i < particles.size(); ++i){
    
    // Step 0, find all the landmarks within the sensor range for each particle
    vector<LandmarkObs> predicted_list;
    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); ++j){
      double px = particles[i].x;
      double py = particles[i].y;
      double lmx = map_landmarks.landmark_list[j].x_f;
      double lmy = map_landmarks.landmark_list[j].x_f;
      double distance = dist(px,py,lmx,lmy);
      if(distance < sensor_range){
        int lmid = map_landmarks.landmark_list[j].id_i;
        predicted_list.push_back(LandmarkObs{lmid,lmx,lmy});
      }      
    }
    
    // Step 1, transform the observations from vehicle coord to map coord
    vector<LandmarkObs> observations_map;// create a new vector for map coordinate
    // extract the particle coord
    double xp = particles[i].x;
    double yp = particles[i].y;
    double thetap = particles[i].theta;
    for(unsigned int k = 1; k < observations.size(); ++ k) {
      double xc = observations[k].x;
      double yc = observations[k].y;
      double id = observations[k].id;
      double xm = xp + xc*cos(thetap) - yc*sin(thetap);
      double ym = yp + xc*sin(thetap) - yc*cos(thetap);
      // assign the transformed coord to a new landmarkObs
      LandmarkObs current_map;
      current_map.id = id;
      current_map.x = xm;
      current_map.y = ym;
      observations_map.push_back(current_map);
    }
    
    // Step 2, set association between the observations and the predictions
    dataAssociation(predicted_list, observations_map);
    
    // Step 3, calculate the particle's final weight
    double sig_x = std_landmark[0];
    double sig_y = std_landmark[1];
    double gauss_norm;
    gauss_norm = 1/(2*M_PI*sig_x*sig_y);
    double exponent;
    
    for(unsigned int m = 0; m < observations_map.size();++m){
      Map::single_landmark_s landmark = map_landmarks.landmark_list[m];
      double mu_x = landmark.x_f;
      double mu_y = landmark.y_f;
      double x_obs = observations_map[m].x;
      double y_obs = observations_map[m].y;
      exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
      double weight_tem;
      weight_tem = gauss_norm * exponent;
      particles[i].weight *= weight_tem;
    }       
    weights.push_back(particles[i].weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  vector<Particle> new_particles;// New blank particles vector 
  new_particles.resize(num_particles);
  // generate random number according to its weight
  random_device rd;
  default_random_engine gen(rd());
  for (int i = 0; i < num_particles; ++ i){
    discrete_distribution<int> index(weights.begin(), weights.end());
    new_particles[i] = particles[index(gen)];
  }
  particles = new_particles;
  
  weights.clear();
  
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations.clear();
  particle.sense_x.clear();
  particle.sense_y.clear();
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