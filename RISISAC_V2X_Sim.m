% import java.math.*
classdef RISISAC_V2X_Sim < handle
    properties
        % System parameters
        fc = 28e9                 % Carrier frequency (28 GHz)
        B = 20e6                  % Bandwidth (20 MHz)
        Ns = 10                   % Number of subcarriers
        Nb = 4                    % Number of BS antennas
        Nt = 4                    % Number of target antennas
        Nr = 64                   % Number of RIS elements
        Nx = 8
        Ny = 8
        Mb = 64
        rate = 0
        c = 3e8;
        lambda = 3e8/28e9;
        sigma_c = sqrt(6.31 * 10^-13);

        h_l = 0;
        h_nl = 0;
        gamma_c = 0;
        SNR = 0;
        cc = 0;
        speed = 10
        end_x = 1000
        Pb = 0
        R_min = 0
        
        % Path loss parameters
        alpha_l = 3.2             % Direct path loss exponent
        alpha_nl = 2.2            % RIS path loss exponent
        rho_l = 3               % Direct path shadow fading
        rho_nl = 4              % RIS path shadow fading
        
        starting_pos = [500,500,0]
        % Locations (in meters)
        bs_loc = [900, 100, 20]   % Base station location
        ris_loc = [200, 300, 40]  % RIS location
        target_loc = [500, 500, 0]% Target location

        stepCount = 0;
        maxSteps = 10000;
        
        car_loc = [500,500,0];
        % Environment dimensions
        env_dims = [1000, 1000]   % Environment dimensions
        destination;
        time = 0;
        arrival_threshold = 10

        % Channel matrices
        H_bt      % Direct channel
        H_br     % BS-RIS channel
        H_rt     % RIS-target channel
        
        phi      % RIS phase shifts
        
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
            obj.calculated_values();
            obj.destination = [randi([990, 1000]), randi([990, 1000]), 0];
        end

        function nb = get_Nb(obj)
            nb = obj.Nb;
        end

        
        function calculated_values(obj)
            K_dB = 4; % Rician K-factor in dB
            K = 10^(K_dB/10);
            sigma = sqrt(1/(2*(K+1)));
            mu = sqrt(K/(K+1));
            
            % Generate small-scale fading
            obj.h_l = (sigma * complex(randn(1,1), randn(1,1))) + mu;
            
            % Generate small-scale fading
            obj.h_nl = (sigma * complex(randn(1,1), randn(1,1))) + mu;
            
            HLos = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nr, obj.Nb);
            HNLos = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb);
            H_combined = HLos + HNLos;

            [Wx,W] = computeWx(obj);
            % disp(W);
            obj.Pb = mean(sum(abs(Wx).^2, 1));
            % disp(['Pb: ' num2str(obj.Pb)]);
            obj.gamma_c = obj.Pb * norm(H_combined * W, 'fro')^2 / obj.sigma_c^2;
            % disp(['gamma_c: ' num2str(obj.gamma_c)]);
            obj.SNR = log10(obj.gamma_c);
            % precoder = eye(obj.Nb) / sqrt(obj.Nb); 

            covariance_matrix = W * W';

            obj.rate = obj.B * log2(1 + obj.gamma_c);
            obj.cc = obj.B * log2(1+obj.SNR);
            obj.R_min = obj.B*60;
        end
        % ! -------------------- CHANNEL INITIALIZATION PART STARTS HERE --------------------        
        function initializeChannels(obj)
            % Pass obj to generate_channels since it needs access to obj.lambda
            [obj.H_bt, obj.H_br, obj.H_rt] = generate_channels(obj, obj.Nt, obj.Nr, obj.Nb);
            % Initialize RIS phase shifts
            rho_r = 1;
            Bit = 2;  

            % Define the phase resolution (covers 0 to 2*pi)
            Delta_delta = 2*pi / (2^Bit);  

            % Create the discrete phase set A
            A = (0 : (2^Bit - 1)) * Delta_delta;

            % Randomly pick each element's phase shift from A
            theta = A(randi(numel(A), obj.Nr, 1));
            u = rho_r * exp(1j*theta);
            obj.phi = diag(u); 
            % obj.phi = eye(obj.Nr);
        end
        
        function [H_bt, H_br, H_rt] = generate_channels(obj, Nt, Nr, Nb)
            % Constants
            d = obj.lambda/2; % antenna spacing
            dr = obj.lambda/2; % element spacing for 2D arrays
            
            % Get geometric parameters
            [~,~,~,~,~,~,~,angles] = obj.computeGeometricParameters();
            
            % Generate BS-Target channel
            H_bt = generate_H_bt(obj, Nt, Nb, angles, obj.lambda, d);
            
            % Generate BS-RIS and RIS-Target channels
            [H_br, H_rt] = generate_H_br_H_rt(obj, Nb, Nr, Nt, angles, obj.lambda, d, dr);
        end

        function H_bt = generate_H_bt(obj, Nt, Nb, angles, lambda, d)
            psi_bt = angles.bs_to_target_transmit;
            psi_tb = angles.bs_to_target_receive;
            
            a_psi_bt = compute_a_psi(obj, Nb, psi_bt, lambda, d);
            a_psi_tb = compute_a_psi(obj, Nt, psi_tb, lambda, d);
            
            H_bt = a_psi_tb * a_psi_bt';
        end

        function [H_br, H_rt] = generate_H_br_H_rt(obj, Nb, Nr, Nt, angles, lambda, d, dr)
            % BS-RIS channel parameters
            psi_br = angles.bs_to_ris.azimuth;
            phi_abr = angles.bs_to_ris.elevation_azimuth;
            phi_ebr = angles.bs_to_ris.elevation_angle;
            
            % RIS-Target channel parameters
            psi_rt = angles.ris_to_target.azimuth;
            phi_art = angles.ris_to_target.elevation_angle;
            phi_ert = angles.ris_to_target.elevation_angle;
            
            % Generate channels
            H_br = generate_H_br(obj, Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, dr, d);
            H_rt = generate_H_rt(obj, Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d);
        end

        function H_br = generate_H_br(obj, Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, dr, d)
            a_psi_br = compute_a_psi(obj, Nb, psi_br, lambda, d);
            a_phi_abr = compute_a_phi(obj, sqrt(Nr), phi_abr, phi_ebr, lambda, dr);
            
            H_br = a_phi_abr * a_psi_br';
        end
        
        function H_rt = generate_H_rt(obj, Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d)
            a_psi_rt = compute_a_psi(obj, Nt, psi_rt, lambda, d);
            a_phi_art = compute_a_phi(obj, sqrt(Nr), phi_art, phi_ert, lambda, dr);
            
            H_rt = a_psi_rt * a_phi_art';
        end

        function a_vec = compute_a_psi(~, Nant, psi, lambda, d)
            k = 2*pi/lambda;
            n = 0:(Nant-1);
            phase_terms = exp(1j * k * d * n * sin(psi));
            a_vec = phase_terms(:) / sqrt(Nant);
        end
        
        function a_phi = compute_a_phi(~, Nx, phi_a, phi_e, lambda, dr)
            N2 = Nx * Nx;
            a_phi = zeros(N2, 1);
            k = 2*pi/lambda;
            
            idx = 1;
            for m = 1:Nx
                for n = 1:Nx
                    phase_term = exp(1j * k * dr * (m * sin(phi_a) * sin(phi_e) + n * cos(phi_e)));
                    a_phi(idx) = phase_term;
                    idx = idx + 1;
                end
            end
            
            a_phi = a_phi / sqrt(N2);
        end

        function H_Los = generate_H_Los(obj, H_bt, Nt, ~, Nb)
            % Parameters
            K_dB = 4; % Rician K-factor in dB
            K = 10^(K_dB/10);
            sigma = sqrt(1/(2*(K+1)));
            mu = sqrt(K/(K+1));
            
            % Generate small-scale fading
            obj.h_l = (sigma * complex(randn(1,1), randn(1,1))) + mu;
            
            % Path loss in linear scale (110 dB)
            obj.rho_l = 3;
            
            % Calculate gamma_l
            gamma_l = sqrt(Nb*Nt)/sqrt(obj.rho_l);
            
            [~,~,~,~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_l = delays.line_of_sight;
            
            % Generate H_Los for all subcarriers with correct dimensions Nt × Nb
            H_Los_3d = zeros(Nt, Nb, obj.Ns);
            
            for n = 1:obj.Ns
                phase = exp(1j*2*pi*obj.B*(n-1)*tau_l/obj.Ns);
                H_Los_3d(:,:,n) = gamma_l * obj.h_l * H_bt * phase;  
            end
            
            % Convert 3D to 2D by averaging over subcarriers
            H_Los = mean(H_Los_3d, 3);  % Nt × Nb matrix
        end
        
        
        function H_NLoS = generate_H_NLoS(obj, H_rt, H_br, Nt, Nr, Nb)
            % Parameters
            K_dB = 4; % Rician K-factor in dB
            K = 10^(K_dB/10);
            sigma = sqrt(1/(2*(K+1)));
            mu = sqrt(K/(K+1));
            
            % Generate small-scale fading
            obj.h_nl = (sigma * complex(randn(1,1), randn(1,1))) + mu;
            
            % Path loss in linear scale
            obj.rho_nl = 4;
            
            % Calculate gamma_nl
            gamma_nl = sqrt(Nb*Nr)/sqrt(obj.rho_nl);
            
            [~,~,~,~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_nl = delays.non_line_of_sight;
            
            % Generate RIS reflection parameters (u)
            rho_r = 1;
            theta = 2*pi*rand(Nr,1);
            u = rho_r * exp(1j*theta);
            obj.phi = diag(u);
            
            % Generate H_NLoS for all subcarriers
            H_NLoS_3d = zeros(Nt, Nb, obj.Ns);
            
            for n = 1:obj.Ns
                phase = exp(1j*2*pi*obj.B*(n-1)*tau_nl/obj.Ns);
                % H_rt(Nt×Nr) * phi(Nr×Nr) * H_br(Nr×Nb)
                H_NLoS_3d(:,:,n) = gamma_nl * obj.h_nl * H_rt * obj.phi * H_br * phase;
            end
            
            % Convert 3D to 2D by averaging over subcarriers
            H_NLoS = mean(H_NLoS_3d, 3);  % Nt × Nb matrix
        end        
        

        % ! -------------------- CHANNEL INITIALIZATION PART ENDS HERE --------------------        
        
        % ! -------------------- MACHINE LEARNING PART STARTS HERE --------------------        
        function state = getState(obj)
            % Construct the state vector for the RL agent as described in the research.
            % State: [Phase shift info (phi), Communication rate (Rc), Channel info (H)]
            
            % Phase shift information - split complex into real and imaginary parts
            phi_real = real(diag(obj.phi))'; % 64 elements
            phi_imag = imag(diag(obj.phi))'; % 64 elements
            
            % Communication capacity (assumed to be a single value for this example)
            % You would need to compute this based on your system dynamics or have it as an object property
            Rc = obj.rate; % Example placeholder
            
            H = obj.H_bt;
            % Channel Information - split complex into real and imaginary parts
            % Assuming obj.H is a complex matrix for the channel info
            H_real = real(H(:))'; % Flatten to a row vector
            H_imag = imag(H(:))'; % Flatten to a row vector
            
            % Construct the full state vector
            state = [
                phi_real, ... % Phase shift real part
                phi_imag, ... % Phase shift imaginary part
                Rc, ... % Communication capacity
                H_real, ... % Channel info real part
                H_imag ... % Channel info imaginary part
            ];
            
            % Ensure it's a row vector
            % disp('Max imaginary part in state:'), max(abs(imag(state(:))))
            state = real(state);
            state = state(:)';
            % disp('State class: '), class(state)
            % disp('Max imaginary part: '), max(abs(imag(state(:))))

        end

        function [next_state, reward, done] = step(obj, action)
            % Process RIS phases from action
            ris_phases = action(1:obj.Nr);
            obj.phi = diag(exp(1j * 2 * pi * ris_phases));
            % [~,W] = computeWx(obj);
            % covar_matrix = W * W';
            [peb] = obj.calculatePerformanceMetrics();
            
            
            % Update the vehicle's position
            direction = (obj.destination - obj.car_loc) / norm(obj.destination - obj.car_loc);
            obj.car_loc = obj.car_loc + direction * obj.speed * obj.dt;
            obj.time = obj.time + obj.dt;
            
            % Update the target location
            obj.target_loc = obj.car_loc;
            
            % Recompute geometric parameters
            [~,~,~,~,~,~,~,~] = computeGeometricParameters(obj);
            
            % Increase step counter
            obj.stepCount = obj.stepCount + 1;
            
            % Get new state
            next_state = getState(obj);
            
            % Calculate reward - consider distance-based component
            reward = obj.computeReward(peb);
            % disp(peb);
            reward = sqrt((real(reward)^2) - (imag(reward)^2));
            % disp(reward)
            
            % Check termination conditions
            destination_reached = norm(obj.car_loc - obj.destination) < obj.arrival_threshold;
            out_of_bounds = checkOutOfBounds(obj);
            timeout = obj.stepCount >= obj.maxSteps;
            
            % Set done flag
            done = destination_reached || out_of_bounds || timeout;
        end

        function out_of_bounds = checkOutOfBounds(obj)
            if obj.car_loc(1) > 1000 || obj.car_loc(2) > 1000
                out_of_bounds = true;
            else
                out_of_bounds = false;
            end
        end

        function reward = computeReward(obj, peb)
            % Compute reward based on (1/PEB) with constraint penalty
            % Parameters
            Q = 0.5; % Reward factor for unsatisfied constraints (ρ in your notation)
                        % You can adjust this value between 0 and 1
            
            % Check if constraints are satisfied
            constraints_satisfied = (obj.rate >= obj.R_min);

            base_reward = 1/peb;
            
            if ~constraints_satisfied
                % Apply penalty factor Q when constraints are not satisfied
                reward = base_reward * Q;
            else
                % Full reward when constraints are satisfied
                reward = base_reward;
            end
        end

        function done = isEpisodeDone(obj)
            % Check if vehicle has reached the destination
            epsilon = 10.0; % Increased threshold to 5.0 meters for more reasonable arrival detection
            reached_dest = norm(obj.car_loc - obj.destination) < epsilon;
            
            % Check if vehicle is out of bounds
            pos = obj.car_loc;
            out_of_bounds = any(pos(1:2) < 0) || ...
                            pos(1) > obj.env_dims(1) || ...
                            pos(2) > obj.env_dims(2);
            
            % Episode ends if the vehicle reaches its destination or goes out of bounds
            done = reached_dest || out_of_bounds;
            
            % Optional: Add debug information
            if done && reached_dest
                disp(['Destination reached with final distance: ' num2str(norm(obj.car_loc - obj.destination))]);
            elseif done && out_of_bounds
                disp('Episode terminated: Out of bounds');
            end
        end
        
        
        function state = reset(obj)
            % Reset simulation state
            obj.initializeChannels();
            obj.calculated_values();
            obj.destination = [randi([990, 1000]), randi([990, 1000]), 0];

            obj.car_loc = obj.starting_pos;
            obj.target_loc = obj.car_loc;
            obj.time = 0;
            obj.stepCount = 0;

            
            % Return the current state
            state = obj.getState();
        end
        % ! -------------------- MACHINE LEARNING PART ENDS HERE --------------------

        % ! -------------------- PEB COMPUTATION PART STARTS HERE --------------------
        

        function [R_min] = computeR_min(obj)
            % Shannon capacity formula for baseline
            R_theoretical = obj.B * log2(1 + obj.gamma_c);

            % Set R_min as a fraction of theoretical maximum
            R_min = 0.5 * R_theoretical;
        end


        function [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj)
            % Extract coordinates
            xb = obj.bs_loc(1);    yb = obj.bs_loc(2);    zb = obj.bs_loc(3);
            xr = obj.ris_loc(1);   yr = obj.ris_loc(2);   zr = obj.ris_loc(3);
            xt = obj.target_loc(1); yt = obj.target_loc(2); zt = obj.target_loc(3);
            
            % Calculate 3D Euclidean distances
            L1 = sqrt((xb - xr)^2 + (yb - yr)^2 + (zb - zr)^2);
            L2 = sqrt((xr - xt)^2 + (yr - yt)^2 + zr^2);
            L3 = sqrt((xb - xt)^2 + (yb - yt)^2 + zb^2);
            
            % Calculate 2D (X-Y plane) projections
            L_proj1 = sqrt((xb - xr)^2 + (yb - yr)^2);
            L_proj2 = sqrt((xr - xt)^2 + (yr - yt)^2);
            L_proj3 = sqrt((xb - xt)^2 + (yb - yt)^2);
            
            % ? there is a delay calculation code here
            % Calculate signal delays
            delays.line_of_sight = L3 / obj.c;
            delays.non_line_of_sight = (L1 + L2) / obj.c;
            
            % Calculate angles
            % BS to RIS angles
            angles.bs_to_ris.azimuth = asin((zb - zr) / L1);
            angles.bs_to_ris.elevation_azimuth = asin((xb - xr) / L_proj1);
            angles.bs_to_ris.elevation_angle = acos((zb - zr) / L1);
            
            % RIS to Target angles
            angles.ris_to_target.aoa = asin(zr / L2);
            angles.ris_to_target.azimuth = acos((yr - yt) / L_proj2);
            angles.ris_to_target.elevation_angle = acos(zr / L2);

            % BS to Target angles
            angles.bs_to_target_transmit = acos(zb/L3);
            angles.bs_to_target_receive = asin(zb/L3);
        end

        function [peb] = calculatePerformanceMetrics(obj)
            % Compute FIM and CRLB
            [J, ~, ~] = computeFisherInformationMatrix(obj);

            % Compute optimized CRLB
            CRLB = inv(J);

            % Calculate final PEB
            peb = sqrt(trace(CRLB));
            rate_constraint_satisfied = (obj.rate >= obj.R_min);

            if ~rate_constraint_satisfied
                % Apply penalty - this is a better formulation
                penalty_factor = 1 + (obj.R_min - obj.rate)/obj.R_min;
                peb = peb * penalty_factor;
            end
            peb = sqrt((real(peb)^2) - (imag(peb)^2));
            if(peb>12)
                peb=12;
            end
        end
        
        function [T] = computeTransformationMatrix(obj)
            T = zeros(2, 7);
            xb = obj.bs_loc(1);    yb = obj.bs_loc(2);    zb = obj.bs_loc(3);
            xr = obj.ris_loc(1);   yr = obj.ris_loc(2);   zr = obj.ris_loc(3);
            xt = obj.target_loc(1); yt = obj.target_loc(2); zt = obj.target_loc(3);

            [~, L2, L3, ~, L_proj2, ~, ~, ~] = computeGeometricParameters(obj);

            T(1,1) = (xt-xb) / (obj.c*L3);
            T(1,2) = (xt-xr) / (obj.c*L2);
            T(1,3) = (zr*(xt-xr)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(1,4) = ((yr-yt)*(xr-xt)) / ((L2^3)*sqrt(1-((yr-yt)^2)/(L2)^2));
            T(1,5) = (zr*(xr-xt)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(1,6) = (zb*(xt-xb)) / ((L3^3)*sqrt(1-L3^2));
            T(1,7) = (zb*(xt-xb)) / ((L3^3)*sqrt(1-L3^2));
            
            T(2,1) = (yt-yb) / (obj.c*L3);
            T(2,2) = (yt-yr) / (obj.c*L2);
            T(2,3) = (zr*(yt-yr)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(2,4) = ((L_proj2^2) * (yr-yt)*(xr-xt)) / ((L2^3)*sqrt(1-((yr-yt)^2)/(L2)^2));
            T(2,5) = (zr*(yr-yt)) / ((L2^3)*sqrt(1-((zr)^2)/(L2)^2));
            T(2,6) = (zb*(yt-yb)) / ((L3^3)*sqrt(1-L3^2));
            T(2,7) = (zb*(yt-yb)) / ((L3^3)*sqrt(1-L3^2));
        end
        
        function [Wx, W] = computeWx(obj)
            N = obj.Ns;    % Number of subcarriers
            
            % Generate beamforming matrix W [Nb x Mb]
            W = rand(obj.Nb, obj.Mb) + 1j*randn(obj.Nb, obj.Mb);
            W = W ./ vecnorm(W);  % Normalize each column
            
            % Generate transmit data x[n] for each subcarrier n
            % x[n] is Mb x 1 complex Gaussian with zero mean and unit variance
            X = (randn(obj.Mb, N) + 1j*randn(obj.Mb, N)) / sqrt(2);  % Division by sqrt(2) ensures unit variance
            
            % Calculate Wx[n] for all subcarriers
            Wx = W * X;  % Results in [Nb x N] matrix where each column is Wx[n] for nth subcarrier
        end
        
        function [J, Jzao, T] = computeFisherInformationMatrix(obj)
            sigma_s = sqrt(obj.SNR/obj.Pb);  % Noise variance (placeholder)
            [T] = computeTransformationMatrix(obj);
            [Wx,~] = computeWx(obj);
            gamma_l = sqrt(obj.Nb*obj.Nt)/sqrt(obj.rho_l);
            gamma_nl = sqrt(obj.Nb*obj.Nt)/sqrt(obj.rho_nl);
            
            [A1, A2, A3, A4] = computeAmplitudeMatrices(obj, obj.Nt, obj.B, gamma_l, gamma_nl, obj.h_l, obj.h_nl);
            [Jzao] = calculateJacobianMatrix(obj, obj.Pb, sigma_s, obj.Nt, Wx, A1, A2, A3, A4);
            
            % Compute final Fisher Information Matrix
            J = T * Jzao * T';
        end
        
        function [A1, A2, A3, A4] = computeAmplitudeMatrices(obj, N, B, gamma_l, gamma_nl, h_l, h_nl)
            [~, ~, ~, ~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            A3 = zeros(1,4);
            A4 = zeros(1,4);
            for n = 1:N
                A3(:,n) = gamma_nl * h_nl * exp(1j * 2 * pi * B * (n/N) * delays.non_line_of_sight);
                A4(:,n) = gamma_l * h_l * exp(1j * 2 * pi * B * (n/N) * delays.line_of_sight);
            end
            A1 = zeros(1,4);
            A2 = zeros(1,4);
            % Calculate A1 and A2 for each n
            for n = 1:N
                A1(:,n) = gamma_l * h_l * 1j * 2 * pi * exp(1j * 2 * pi * B * (n/N) * delays.line_of_sight);
                A2(:,n) = gamma_nl * h_nl * 1j * 2 * pi * exp(1j * 2 * pi * B * (n/N) * delays.non_line_of_sight);
            end
        end 

        function [J_zao] = calculateJacobianMatrix(obj, Pb, sigma, N, Wx, A1, A2, A3, A4)
            % Initialize 7x7 Jacobian matrix
            J_zao = zeros(7, 7);
            [~, ~, ~, ~, ~, ~, ~, angles] = computeGeometricParameters(obj);
            psi_rt = angles.ris_to_target.aoa;
            psi_bt = angles.bs_to_target_transmit;
            psi_tb = angles.bs_to_target_receive;
            psi_br = angles.bs_to_ris.azimuth;
            phi_rt_a = angles.ris_to_target.azimuth;
            phi_rt_e = angles.ris_to_target.elevation_angle;
            phi_br_a = angles.bs_to_ris.elevation_azimuth;
            phi_br_e = angles.bs_to_ris.elevation_angle;

            indices = 1:(obj.Nt);
            a_rt = 1j * (2 * pi / obj.lambda) * cos(psi_rt) * diag(indices);
            indices = 1:(obj.Nb);
            a_bt = 1j * (2 * pi / obj.lambda) * cos(psi_bt) * diag(indices);
            indices = 1:(obj.Nt);
            a_tb = 1j * (2 * pi / obj.lambda) * cos(psi_bt) * diag(indices);

            a_rt_a = 1j * (2 * pi / obj.lambda) * obj.lambda/2 * ((obj.Nx-1) * cos(phi_rt_a) * sin(phi_rt_e));
            a_rt_e = 1j * (2 * pi / obj.lambda) * obj.lambda/2 * (((obj.Nx-1) * sin(phi_rt_a) * cos(phi_rt_e)) - ((obj.Ny-1) * sin(phi_rt_e)));
            
            % TODO: implement all the a vector
            a_vec = compute_a_psi(obj, obj.Nt, psi_bt, obj.lambda, obj.lambda/2);
            a_psi_bt = a_vec;
            a_vec = compute_a_psi(obj, obj.Nt, psi_tb, obj.lambda, obj.lambda/2);
            a_psi_tb = a_vec;
            a_vec = compute_a_psi(obj, obj.Nt, psi_rt, obj.lambda, obj.lambda/2);
            a_psi_rt = a_vec;
            a_vec = compute_a_psi(obj, obj.Nb, psi_br, obj.lambda, obj.lambda/2);
            a_psi_br = a_vec;


            % TODO: implement all the steering vector
            a_phi_br = compute_a_phi(obj, obj.Nx, phi_br_a, phi_br_e, obj.lambda, obj.lambda/2);
            a_phi_rt = compute_a_phi(obj, obj.Nx, phi_rt_a, phi_rt_e, obj.lambda, obj.lambda/2);

            % disp('Intermediate dimensions:')
            % disp(size(A1 * a_psi_bt))
            % disp(size(a_psi_tb' * Wx))
            % disp(size(a_psi_bt * (a_psi_tb' * Wx)))

            % Calculate partial derivatives for each n
            for n = 1:N
                % Calculate all partial derivatives
                d_mu_array = cell(7, 1);
                
                % Partial derivatives with respect to each parameter
                d_mu_array{1} = (A1 * a_psi_bt) * (a_psi_tb' * Wx(:,n));  % d_mu_d_tau
                
                d_mu_array{2} = (A2 * a_psi_rt) * ...
                    (a_phi_rt' * ...          % d_mu_d_tau_rt
                    obj.phi * a_phi_br) * ...
                    (a_psi_br' * ...
                    Wx(:,n));
                
                d_mu_array{3} = (A3 * a_rt * a_psi_rt) * ...
                    (a_phi_rt' * ...          % d_mu_d_psi_rt
                    obj.phi * a_phi_br) * ...
                    (a_psi_br' * Wx(:,n));
                
                d_mu_array{4} = (A3 * a_psi_rt) * ...
                    (a_phi_rt' * ...          % d_mu_d_phi_a
                    diag(a_rt_a) * ...
                    obj.phi * a_phi_br) * ...
                    (a_psi_br' * Wx(:,n));
                
                d_mu_array{5} = (A3 * a_psi_rt) * ...
                    (a_phi_rt' * ...          % d_mu_d_phi_e
                    diag(a_rt_e) * ...
                    obj.phi * a_phi_br)* ...
                        (a_psi_br' * Wx(:,n));
                
                d_mu_array{6} = (A4 * a_bt' * a_psi_bt) * ...   % d_mu_d_psi_br
                    (a_psi_tb' * Wx(:,n));
            
                d_mu_array{7} = (A4 * a_tb * a_psi_bt) * ... 
                    (a_psi_tb' * ...   % d_mu_d_psi_bt
                    Wx(:,n));
                
                % Calculate Jacobian matrix elements
                for i = 1:7
                    for j = 1:7

                        J_zao(i,j) = J_zao(i,j) + real(d_mu_array{i}' * d_mu_array{j});
                    end
                end
            end
            % Apply scaling factor
            % J_zao = (2 * Pb / (sigma^2)) * J_zao;
            
        end
        % ! -------------------- PEB COMPUTATION PART ENDS HERE --------------------        

    end
    
end
