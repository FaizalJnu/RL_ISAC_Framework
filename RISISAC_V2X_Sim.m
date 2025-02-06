classdef RISISAC_V2X_Sim < handle
    properties
        % System parameters
        fc = 28e9                 % Carrier frequency (28 GHz)
        B = 20e6                  % Bandwidth (20 MHz)
        Ns = 10                   % Number of subcarriers
        Nb = 4                    % Number of BS antennas
        Nt = 4                    % Number of target antennas
        Nr = 64                   % Number of RIS elements
        
        % Path loss parameters
        alpha_l = 3.2             % Direct path loss exponent
        alpha_nl = 2.2            % RIS path loss exponent
        sigma_l = 3               % Direct path shadow fading
        sigma_nl = 4              % RIS path shadow fading
        
        % Locations (in meters)
        bs_loc = [900, 100, 20]   % Base station location
        ris_loc = [200, 300, 40]  % RIS location
        target_loc = [500, 500, 0]% Target location
        
        % Environment dimensions
        env_dims = [1000, 1000]   % Environment dimensions
        
        % Channel matrices
        H_d      % Direct channel
        H_br     % BS-RIS channel
        H_rt     % RIS-target channel
        Phi      % RIS phase shifts
        
        % Vehicle dynamics parameters
        vehicle = struct(...
            'position', [0, 0, 0], ...          % [x, y, z] in meters
            'velocity', [0, 0, 0], ...          % [vx, vy, vz] in m/s
            'acceleration', [0, 0, 0], ...      % [ax, ay, az] in m/s^2
            'heading', 0, ...                   % in radians
            'mass', 1500, ...                   % in kg
            'max_speed', 30, ...                % in m/s
            'max_acceleration', 3, ...          % in m/s^2
            'max_deceleration', -6, ...         % in m/s^2
            'max_steering_angle', pi/4, ...     % in radians
            'wheelbase', 2.7 ...                % in meters
        )
        
        % Simulation parameters
        dt = 0.1  % Time step in seconds
    end
    
    methods
        function obj = RISISAC_V2X_Sim()
            % Initialize channels
            obj.initializeChannels();
        end

        function nb = get_Nb(obj)
            nb = obj.Nb;
        end
        
        function initializeChannels(obj)
            % Initialize direct channel
            obj.H_d = obj.generateDirectChannel();
            
            % Initialize BS-RIS channel
            obj.H_br = obj.generateBSRISChannel();
            
            % Initialize RIS-target channel
            obj.H_rt = obj.generateRISTargetChannel();
            
            % Initialize RIS phase shifts
            obj.Phi = eye(obj.Nr);
        end
        
        function H = generateDirectChannel(obj)
            % Generate direct channel with path loss and shadow fading
            d = norm(obj.bs_loc - obj.target_loc);
            PL = 20*log10(4*pi*d*obj.fc/3e8) + 10*obj.alpha_l*log10(d);
            shadow = obj.sigma_l * randn(1);
            
            % Generate Rician fading channel
            K_factor = 10;  % Rician K-factor
            H = sqrt(1/(1+K_factor)) * (sqrt(1/2)*(randn(obj.Nt,obj.Nb) + 1j*randn(obj.Nt,obj.Nb))) + ...
                sqrt(K_factor/(1+K_factor)) * ones(obj.Nt,obj.Nb);
            
            % Apply path loss and shadow fading
            H = H * 10^(-(PL + shadow)/20);
        end
        
        function H = generateBSRISChannel(obj)
            % Generate BS-RIS channel with path loss
            d = norm(obj.bs_loc - obj.ris_loc);
            PL = 20*log10(4*pi*d*obj.fc/3e8) + 10*obj.alpha_nl*log10(d);
            shadow = obj.sigma_nl * randn(1);
            
            % Generate channel matrix
            H = sqrt(1/2)*(randn(obj.Nr,obj.Nb) + 1j*randn(obj.Nr,obj.Nb));
            
            % Apply path loss and shadow fading
            H = H * 10^(-(PL + shadow)/20);
        end
        
        function H = generateRISTargetChannel(obj)
            % Generate RIS-target channel with path loss
            d = norm(obj.ris_loc - obj.target_loc);
            PL = 20*log10(4*pi*d*obj.fc/3e8) + 10*obj.alpha_nl*log10(d);
            shadow = obj.sigma_nl * randn(1);
            
            % Generate channel matrix
            H = sqrt(1/2)*(randn(obj.Nt,obj.Nr) + 1j*randn(obj.Nt,obj.Nr));
            
            % Apply path loss and shadow fading
            H = H * 10^(-(PL + shadow)/20);
        end
        
        function updateVehicleDynamics(obj, throttle, steering_angle)
            % Constrain inputs
            throttle = max(min(throttle, 1), -1);  % Normalize between -1 and 1
            steering_angle = max(min(steering_angle, obj.vehicle.max_steering_angle), ...
                               -obj.vehicle.max_steering_angle);
            
            % Calculate acceleration based on throttle
            if throttle >= 0
                target_accel = throttle * obj.vehicle.max_acceleration;
            else
                target_accel = throttle * abs(obj.vehicle.max_deceleration);
            end
            
            % Update heading based on steering angle and velocity
            speed = norm(obj.vehicle.velocity(1:2));
            if speed > 0.1  % Only update heading when moving
                turning_radius = obj.vehicle.wheelbase / tan(steering_angle);
                angular_velocity = speed / turning_radius;
                obj.vehicle.heading = obj.vehicle.heading + angular_velocity * obj.dt;
                % Normalize heading to [-pi, pi]
                obj.vehicle.heading = mod(obj.vehicle.heading + pi, 2*pi) - pi;
            end
            
            % Update acceleration, velocity, and position
            % Decompose acceleration into x and y components based on heading
            obj.vehicle.acceleration(1) = target_accel * cos(obj.vehicle.heading);
            obj.vehicle.acceleration(2) = target_accel * sin(obj.vehicle.heading);
            
            % Update velocity using acceleration
            new_velocity = obj.vehicle.velocity + obj.vehicle.acceleration * obj.dt;
            
            % Constrain speed to maximum
            current_speed = norm(new_velocity(1:2));
            if current_speed > obj.vehicle.max_speed
                new_velocity(1:2) = new_velocity(1:2) * obj.vehicle.max_speed / current_speed;
            end
            obj.vehicle.velocity = new_velocity;
            
            % Update position
            obj.vehicle.position = obj.vehicle.position + obj.vehicle.velocity * obj.dt;
        end
        
        function state = getState(obj)
            % Construct state vector for RL agent
            % Format: [position(3), velocity(3), acceleration(3), heading(1), 
            %          ris_phases_real(64), ris_phases_imag(64)]
            state = [
                obj.vehicle.position(:)', ...           % 3 elements
                obj.vehicle.velocity(:)', ...           % 3 elements
                obj.vehicle.acceleration(:)', ...       % 3 elements
                obj.vehicle.heading, ...                % 1 element
                real(diag(obj.Phi))', ...              % 64 elements
                imag(diag(obj.Phi))'                   % 64 elements
            ];
            
            % Ensure it's a row vector
            state = state(:)';
        end
        
        function [next_state, reward, done] = step(obj, action)
            % Parse action vector
            ris_phases = action(1:obj.Nr);           % RIS phase shifts
            throttle = action(obj.Nr + 1);           % Vehicle throttle
            steering = action(obj.Nr + 2);           % Steering angle
            
            % Update RIS phase shifts
            obj.Phi = diag(exp(1j * 2 * pi * ris_phases));
            
            % Update vehicle state
            obj.updateVehicleDynamics(throttle, steering);
            
            % Calculate performance metrics
            precoder = eye(obj.Nb) / sqrt(obj.Nb);  % Simple precoder

            [rate, peb] = obj.calculatePerformanceMetrics(precoder);
            
            % Calculate reward
            reward = obj.calculateReward(rate, peb);
            
            % Get next state
            next_state = obj.getState();
            
            % Check if episode is done
            done = obj.isEpisodeDone();
        end

        function [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj)
            % Extract coordinates
            xb = obj.bs_loc(1);    yb = obj.bs_loc(2);    zb = obj.bs_loc(3);
            xr = obj.ris_loc(1);   yr = obj.ris_loc(2);   zr = obj.ris_loc(3);
            xt = obj.target_loc(1); yt = obj.target_loc(2); zt = obj.target_loc(3);
            
            % Speed of light
            c = 3e8;
            
            % Calculate 3D Euclidean distances
            L1 = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2);
            L2 = sqrt((xr - xt)^2 + (yr - yt)^2 + zr^2);
            L3 = sqrt((xb - xt)^2 + (yb - yt)^2 + zb^2);
            
            % Calculate 2D (X-Y plane) projections
            L_proj1 = sqrt((xb - xr)^2 + (yb - yr)^2);
            L_proj2 = sqrt((xr - xt)^2 + (yr - yt)^2);
            L_proj3 = sqrt((xb - xt)^2 + (yb - yt)^2);
            
            % Calculate signal delays
            delays.line_of_sight = L3 / c;
            delays.non_line_of_sight = (L1 + L2) / c;
            
            % Calculate angles
            % BS to RIS angles
            angles.bs_to_ris.azimuth = asin((zb - zr) / L1);
            angles.bs_to_ris.elevation_azimuth = asin((xb - xr) / L_proj1);
            angles.bs_to_ris.elevation_angle = acos((zb - zr) / L1);
            
            % RIS to Target angles
            angles.ris_to_target.aoa = asin(zr / L2);
            angles.ris_to_target.azimuth = acos((yr - yt) / L_proj2);
            angles.ris_to_target.elevation_angle = acos(zr / L2);
        end
        
        function [rate, peb, additionalMetrics] = calculatePerformanceMetrics(obj, precoder)
            % Compute geometric parameters
            [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj);
            
            % Calculate effective channel
            H_eff = obj.H_d + obj.H_rt * obj.Phi * obj.H_br;
            
            % SNR calculation
            SNR = 20; % dB
            gamma = 10^(SNR/10);
            
            % Rate calculation (closer to the problem formulation)
            rate = log2(1 + gamma * det(eye(obj.Nt) + H_eff * (precoder * precoder') * H_eff') / trace(H_eff * (precoder * precoder') * H_eff'));
            
            % Compute Transformation Matrix and Fisher Information Matrix
            [T, dParams] = computeTransformationMatrix(obj);
            
            % Detailed Fisher Information Matrix computation
            [J, Jzao, T] = computeFisherInformationMatrix(obj, precoder, H_eff);
                
            % Position Error Bound calculation
            CRLB = inv(J);

            peb = sqrt(trace(CRLB(1:2, 1:2))); % Focus on x-y positioning

            % try
            %     [J, Jzao, T] = computeFisherInformationMatrix(obj, precoder, H_eff);
                
            %     % Position Error Bound calculation
            %     CRLB = inv(J);

            %     peb = sqrt(trace(CRLB(1:2, 1:2))); % Focus on x-y positioning
            % catch
            %     % Fallback to previous simple FIM calculation
            %     % Ensure F is the same size as H_eff * (precoder * precoder') * H_eff'
            %     F = zeros(size(H_eff * (precoder * precoder') * H_eff'));
                
            %     for k = 1:obj.Ns
            %         H_k = H_eff; % Consider frequency-dependent channel
            %         J_k = H_k * (precoder * precoder') * H_k';
                    
            %         % Ensure J_k matches the size of F before addition
            %         if size(J_k) ~= size(F)
            %             warning('Size mismatch in Fisher Information Matrix calculation');
            %             J_k = zeros(size(F)); % Fallback to zero matrix if sizes don't match
            %         end
                    
            %         F = F + real(J_k);
            %     end
                
            %     peb = sqrt(trace(inv(F)));
            % end
            
            % Additional performance metrics
            additionalMetrics = struct(...
                'Distances', struct(...
                    'L1', L1, ...
                    'L2', L2, ...
                    'L3', L3, ...
                    'L_proj1', L_proj1, ...
                    'L_proj2', L_proj2, ...
                    'L_proj3', L_proj3 ...
                ), ...
                'Delays', delays, ...
                'Angles', angles, ...
                'SNR', SNR, ...
                'Gamma', gamma ...
            );
        end
        
        function [T, dParams] = computeTransformationMatrix(obj)
            % Compute initial geometric parameters
            [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj);
            
            % Initialize transformation matrix
            T = zeros(2, 7);
            
            % Structure to store derivatives of parameters
            dParams = struct();
            
            % Compute derivatives with respect to x and y
            [dParams.L1_dx, dParams.L1_dy] = computeDistanceDerivatives(obj, 'L1');
            [dParams.L2_dx, dParams.L2_dy] = computeDistanceDerivatives(obj, 'L2');
            [dParams.L3_dx, dParams.L3_dy] = computeDistanceDerivatives(obj, 'L3');
            
            % Compute derivatives of time delays
            [dParams.tau_l_dx, dParams.tau_l_dy] = computeTimeDelayDerivatives(obj, L3, 'line_of_sight');
            [dParams.tau_nl_dx, dParams.tau_nl_dy] = computeTimeDelayDerivatives(obj, L1, L2, 'non_line_of_sight');
            
            % Compute derivatives of various angles
            [dParams.psi_rt_dx, dParams.psi_rt_dy] = computeAngleDerivatives(obj, 'ris_to_target_aoa');
            [dParams.phi_a_rt_dx, dParams.phi_a_rt_dy] = computeAngleDerivatives(obj, 'ris_to_target_azimuth');
            [dParams.phi_e_rt_dx, dParams.phi_e_rt_dy] = computeAngleDerivatives(obj, 'ris_to_target_elevation');
            [dParams.psi_bt_dx, dParams.psi_bt_dy] = computeAngleDerivatives(obj, 'bs_to_target_transmit');
            [dParams.psi_tb_dx, dParams.psi_tb_dy] = computeAngleDerivatives(obj, 'bs_to_target_receive');
            
            % Populate transformation matrix T according to the paper's formulation
            % T is a 2x7 matrix of partial derivatives with respect to x and y
            T = [
                dParams.tau_l_dx,     dParams.tau_nl_dx,     dParams.psi_rt_dx,     dParams.phi_a_rt_dx,     dParams.phi_e_rt_dx,     dParams.psi_bt_dx,     dParams.psi_tb_dx;
                dParams.tau_l_dy,     dParams.tau_nl_dy,     dParams.psi_rt_dy,     dParams.phi_a_rt_dy,     dParams.phi_e_rt_dy,     dParams.psi_bt_dy,     dParams.psi_tb_dy
            ];
        end
        
        function [dx, dy] = computeDistanceDerivatives(obj, distanceType)
            % Numerical differentiation of distances
            epsilon = 1e-8;  % Small perturbation
            
            % Original location
            orig_loc = obj.target_loc;
            
            % Compute derivatives using central difference method
            dx_perturb_pos = orig_loc;
            dx_perturb_pos(1) = orig_loc(1) + epsilon;
            dx_perturb_neg = orig_loc;
            dx_perturb_neg(1) = orig_loc(1) - epsilon;
            
            dy_perturb_pos = orig_loc;
            dy_perturb_pos(2) = orig_loc(2) + epsilon;
            dy_perturb_neg = orig_loc;
            dy_perturb_neg(2) = orig_loc(2) - epsilon;
            
            % Compute distances
            switch distanceType
                case 'L1'  % BS to RIS distance
                    dist_orig = norm(obj.bs_loc - obj.ris_loc);
                    dist_dx_pos = norm(obj.bs_loc - [dx_perturb_pos(1), obj.ris_loc(2), obj.ris_loc(3)]);
                    dist_dx_neg = norm(obj.bs_loc - [dx_perturb_neg(1), obj.ris_loc(2), obj.ris_loc(3)]);
                    dist_dy_pos = norm(obj.bs_loc - [obj.ris_loc(1), dy_perturb_pos(2), obj.ris_loc(3)]);
                    dist_dy_neg = norm(obj.bs_loc - [obj.ris_loc(1), dy_perturb_neg(2), obj.ris_loc(3)]);
                
                case 'L2'  % RIS to Target distance
                    dist_orig = norm(obj.ris_loc - obj.target_loc);
                    dist_dx_pos = norm(obj.ris_loc - [dx_perturb_pos(1), obj.target_loc(2), obj.target_loc(3)]);
                    dist_dx_neg = norm(obj.ris_loc - [dx_perturb_neg(1), obj.target_loc(2), obj.target_loc(3)]);
                    dist_dy_pos = norm(obj.ris_loc - [obj.target_loc(1), dy_perturb_pos(2), obj.target_loc(3)]);
                    dist_dy_neg = norm(obj.ris_loc - [obj.target_loc(1), dy_perturb_neg(2), obj.target_loc(3)]);
                
                case 'L3'  % BS to Target distance
                    dist_orig = norm(obj.bs_loc - obj.target_loc);
                    dist_dx_pos = norm(obj.bs_loc - [dx_perturb_pos(1), obj.target_loc(2), obj.target_loc(3)]);
                    dist_dx_neg = norm(obj.bs_loc - [dx_perturb_neg(1), obj.target_loc(2), obj.target_loc(3)]);
                    dist_dy_pos = norm(obj.bs_loc - [obj.target_loc(1), dy_perturb_pos(2), obj.target_loc(3)]);
                    dist_dy_neg = norm(obj.bs_loc - [obj.target_loc(1), dy_perturb_neg(2), obj.target_loc(3)]);
            end
            
            % Central difference method for derivatives
            dx = (dist_dx_pos - dist_dx_neg) / (2 * epsilon);
            dy = (dist_dy_pos - dist_dy_neg) / (2 * epsilon);
        end
        
        function [dx, dy] = computeTimeDelayDerivatives(obj, varargin)
            % Numerical differentiation of time delays
            epsilon = 1e-8;  % Small perturbation
            c = 3e8;  % Speed of light
            
            % Original location
            orig_loc = obj.target_loc;
            
            % Compute derivatives using central difference method
            dx_perturb_pos = orig_loc;
            dx_perturb_pos(1) = orig_loc(1) + epsilon;
            dx_perturb_neg = orig_loc;
            dx_perturb_neg(1) = orig_loc(1) - epsilon;
            
            dy_perturb_pos = orig_loc;
            dy_perturb_pos(2) = orig_loc(2) + epsilon;
            dy_perturb_neg = orig_loc;
            dy_perturb_neg(2) = orig_loc(2) - epsilon;
            
            % Handle different delay scenarios
            if length(varargin) == 2 && strcmp(varargin{2}, 'line_of_sight')
                % Line of sight delay (L3/c)
                L3_orig = norm(obj.bs_loc - orig_loc);
                tau_l_orig = L3_orig / c;
                
                L3_dx_pos = norm(obj.bs_loc - dx_perturb_pos);
                L3_dx_neg = norm(obj.bs_loc - dx_perturb_neg);
                L3_dy_pos = norm(obj.bs_loc - dy_perturb_pos);
                L3_dy_neg = norm(obj.bs_loc - dy_perturb_neg);
                
                dx = (L3_dx_pos/c - L3_dx_neg/c) / (2 * epsilon);
                dy = (L3_dy_pos/c - L3_dy_neg/c) / (2 * epsilon);
            
            elseif length(varargin) == 3 && strcmp(varargin{3}, 'non_line_of_sight')
                % Non-line of sight delay (L1 + L2)/c
                L1_orig = norm(obj.bs_loc - obj.ris_loc);
                L2_orig = norm(obj.ris_loc - orig_loc);
                tau_nl_orig = (L1_orig + L2_orig) / c;
                
                % Compute perturbed distances
                L1_dx_pos = norm(obj.bs_loc - obj.ris_loc);
                L2_dx_pos = norm(obj.ris_loc - dx_perturb_pos);
                L1_dx_neg = norm(obj.bs_loc - obj.ris_loc);
                L2_dx_neg = norm(obj.ris_loc - dx_perturb_neg);
                
                L1_dy_pos = norm(obj.bs_loc - obj.ris_loc);
                L2_dy_pos = norm(obj.ris_loc - dy_perturb_pos);
                L1_dy_neg = norm(obj.bs_loc - obj.ris_loc);
                L2_dy_neg = norm(obj.ris_loc - dy_perturb_neg);
                
                dx = ((L1_dx_pos + L2_dx_pos)/c - (L1_dx_neg + L2_dx_neg)/c) / (2 * epsilon);
                dy = ((L1_dy_pos + L2_dy_pos)/c - (L1_dy_neg + L2_dy_neg)/c) / (2 * epsilon);
            else
                error('Invalid time delay computation');
            end
        end
        
        function [dx, dy] = computeAngleDerivatives(obj, angleType)
            % Numerical differentiation of angles
            epsilon = 1e-8;  % Small perturbation
            
            % Original location
            orig_loc = obj.target_loc;
            
            % Compute derivatives using central difference method
            dx_perturb_pos = orig_loc;
            dx_perturb_pos(1) = orig_loc(1) + epsilon;
            
            dy_perturb_pos = orig_loc;
            dy_perturb_pos(2) = orig_loc(2) + epsilon;
            
            % Compute angles based on different types
            switch angleType
                case 'ris_to_target_aoa'
                    % Angle of Arrival at RIS-target link
                    dx = (obj.computeAngleDifference(obj.ris_loc, dx_perturb_pos, 'aoa') - ...
                          obj.computeAngleDifference(obj.ris_loc, orig_loc, 'aoa')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.ris_loc, dy_perturb_pos, 'aoa') - ...
                          obj.computeAngleDifference(obj.ris_loc, orig_loc, 'aoa')) / epsilon;
                
                case 'ris_to_target_azimuth'
                    dx = (obj.computeAngleDifference(obj.ris_loc, dx_perturb_pos, 'azimuth') - ...
                          obj.computeAngleDifference(obj.ris_loc, orig_loc, 'azimuth')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.ris_loc, dy_perturb_pos, 'azimuth') - ...
                          obj.computeAngleDifference(obj.ris_loc, orig_loc, 'azimuth')) / epsilon;
                
                case 'ris_to_target_elevation'
                    dx = (obj.computeAngleDifference(obj.ris_loc, dx_perturb_pos, 'elevation') - ...
                          obj.computeAngleDifference(obj.ris_loc, orig_loc, 'elevation')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.ris_loc, dy_perturb_pos, 'elevation') - ...
                          obj.computeAngleDifference(obj.ris_loc, orig_loc, 'elevation')) / epsilon;
                
                case 'bs_to_target_transmit'
                    dx = (obj.computeAngleDifference(obj.bs_loc, dx_perturb_pos, 'transmit') - ...
                          obj.computeAngleDifference(obj.bs_loc, orig_loc, 'transmit')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.bs_loc, dy_perturb_pos, 'transmit') - ...
                          obj.computeAngleDifference(obj.bs_loc, orig_loc, 'transmit')) / epsilon;
                
                case 'bs_to_target_receive'
                    dx = (obj.computeAngleDifference(obj.bs_loc, dx_perturb_pos, 'receive') - ...
                          obj.computeAngleDifference(obj.bs_loc, orig_loc, 'receive')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.bs_loc, dy_perturb_pos, 'receive') - ...
                          obj.computeAngleDifference(obj.bs_loc, orig_loc, 'receive')) / epsilon;
                
                otherwise
                    error('Invalid angle type');
            end
        end
        
        function angle_diff = computeAngleDifference(obj, ref_loc, target_loc, angleType)
            % Ensure inputs are column vectors
            ref_loc = ref_loc(:);
            target_loc = target_loc(:);
            
            % Pad with zeros if needed to ensure 3D
            if length(ref_loc) < 3
                ref_loc(3) = 0;
            end
            if length(target_loc) < 3
                target_loc(3) = 0;
            end
            
            % Compute angle differences using different methods
            switch angleType
                case 'aoa'
                    % Angle of Arrival
                    dist = norm(ref_loc - target_loc);
                    if dist == 0
                        angle_diff = 0;
                    else
                        angle_diff = asin(ref_loc(3) / dist);
                    end
                
                case 'azimuth'
                    % Azimuth angle
                    dist_2d = norm(ref_loc(1:2) - target_loc(1:2));
                    if dist_2d == 0
                        angle_diff = 0;
                    else
                        angle_diff = acos((ref_loc(2) - target_loc(2)) / dist_2d);
                    end
                
                case 'elevation'
                    % Elevation angle
                    dist = norm(ref_loc - target_loc);
                    if dist == 0
                        angle_diff = 0;
                    else
                        angle_diff = acos(ref_loc(3) / dist);
                    end
                
                case 'transmit'
                    % Transmit angle
                    dist = norm(ref_loc - target_loc);
                    if dist == 0
                        angle_diff = 0;
                    else
                        angle_diff = acos(ref_loc(3) / dist);
                    end
                
                case 'receive'
                    % Receive angle
                    dist = norm(ref_loc - target_loc);
                    if dist == 0
                        angle_diff = 0;
                    else
                        angle_diff = asin(ref_loc(3) / dist);
                    end
                
                otherwise
                    error('Invalid angle type');
            end
        end
        
        function [J, Jzao, T] = computeFisherInformationMatrix(obj, precoder, H_eff)
            % Parameters
            Pb = 1;  % Transmit power (placeholder)
            sigma_s = 1;  % Noise variance (placeholder)
            B = obj.B;  % Bandwidth
            N = obj.Ns;  % Number of subcarriers
            lambda = 3e8 / obj.fc;  % Wavelength
            
            % Compute geometric parameters
            [~, ~, ~, ~, ~, ~, delays, angles] = computeGeometricParameters(obj);
            
            % Estimated parameter vector
            zeta = [
                delays.line_of_sight; 
                delays.non_line_of_sight; 
                angles.ris_to_target.aoa; 
                angles.ris_to_target.azimuth; 
                angles.ris_to_target.elevation_angle; 
                computeBSTargetAngles(obj)
            ];
            
            % Initialize Jzao matrix
            Jzao = zeros(7, 7);
            
            % Compute transformation matrix T
            [T, dParams] = computeTransformationMatrix(obj);
            
            % Calculate dimensions of J_k before the loop
            H_k = H_eff;  % In practice, this might vary with frequency
            J_k_temp = H_k * (precoder * precoder') * H_k';
            [J_k_rows, J_k_cols] = size(J_k_temp);
            
            % Create a mapping matrix to handle dimension mismatch
            mapping_matrix = zeros(7, max(J_k_rows, J_k_cols));
            mapping_matrix(1:min(7, J_k_rows), 1:min(7, J_k_cols)) = 1;
            
            % Compute Jzao using subcarrier-based approach
            for n = 1:N
                % Effective channel for this subcarrier
                H_k = H_eff;  % In practice, this might vary with frequency
                
                % Compute local Fisher Information Matrix contribution
                J_k = H_k * (precoder * precoder') * H_k';
                
                % Accumulate contributions with dimension handling
                for i = 1:7
                    for j = 1:7
                        if i <= size(J_k, 1) && j <= size(J_k, 2)
                            Jzao(i,j) = Jzao(i,j) + 2*Pb/sigma_s * real(J_k(i,j));
                        else
                            % For indices beyond J_k dimensions, add zero contribution
                            Jzao(i,j) = Jzao(i,j) + 0;
                        end
                    end
                end
            end
            
            % Compute final Fisher Information Matrix
            J = T * Jzao * T';
        end
        
        function [psibt, psitb] = computeBSTargetAngles(obj)
            % Compute BS-target transmitting and receiving angles
            L3 = norm(obj.bs_loc - obj.target_loc);
            zb = obj.bs_loc(3);
            
            psibt = acos(zb / L3);
            psitb = asin(zb / L3);
        end
        
        function reward = calculateReward(obj, rate, peb)
            % Calculate reward based on rate, PEB, and vehicle state
            w1 = 0.6;  % Weight for rate
            w2 = 0.3;  % Weight for PEB
            w3 = 0.1;  % Weight for vehicle dynamics
            
            % Normalize metrics
            norm_rate = rate / 10;  % Assuming max rate is 10 bps/Hz
            norm_peb = min(1, peb / 10);  % Assuming max PEB is 10m
            
            % Calculate vehicle dynamics component
            speed = norm(obj.vehicle.velocity(1:2));
            speed_reward = speed / obj.vehicle.max_speed;  % Reward for maintaining good speed
            
            reward = w1 * norm_rate - w2 * norm_peb + w3 * speed_reward;
        end
        
        function done = isEpisodeDone(obj)
            % Check if vehicle is out of bounds
            pos = obj.vehicle.position;
            done = any(pos(1:2) < 0) || ...
                   pos(1) > obj.env_dims(1) || ...
                   pos(2) > obj.env_dims(2);
        end
        
        function state = reset(obj)
            % Reset simulation state
            obj.initializeChannels();
            
            % Reset vehicle state
            obj.vehicle.position = [0, 0, 0];
            obj.vehicle.velocity = [0, 0, 0];
            obj.vehicle.acceleration = [0, 0, 0];
            obj.vehicle.heading = 0;
            
            % Return the current state
            state = obj.getState();
            
        end

    end
    
end

