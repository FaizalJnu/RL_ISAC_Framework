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
        
        % Path loss parameters
        alpha_l = 3.2             % Direct path loss exponent
        alpha_nl = 2.2            % RIS path loss exponent
        rho_l = 3               % Direct path shadow fading
        rho_nl = 4              % RIS path shadow fading
        
        % Locations (in meters)
        bs_loc = [900, 100, 20]   % Base station location
        ris_loc = [200, 300, 40]  % RIS location
        target_loc = [500, 500, 0]% Target location

        stepCount = 0;
        maxSteps = 10000;
        
        car_loc = target_loc;
        % Environment dimensions
        env_dims = [1000, 1000]   % Environment dimensions
        destination = [randi([0, simulation_bounds(1)]), randi([0, simulation_bounds(2)]), 0];
        time = 0;
        
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
            gamma_c = obj.Pb * norm(H_combined * W, 'fro')^2 / obj.sigma_c^2;
            obj.gamma_c = gamma_c / 1e8;
            % disp(['gamma_c: ' num2str(obj.gamma_c)]);
            obj.SNR = log10(obj.gamma_c);
            % precoder = eye(obj.Nb) / sqrt(obj.Nb); 

            covariance_matrix = W * W';

            obj.rate = obj.B * log2(1 + obj.gamma_c * det(eye(obj.Nt) + H_combined * covariance_matrix * H_combined') / ...
            trace(H_combined * covariance_matrix * H_combined'));
            obj.cc = obj.B * log2(1+obj.SNR);
        end
        % ! -------------------- CHANNEL INITIALIZATION PART STARTS HERE --------------------        
        function initializeChannels(obj)
            Nr = obj.Nr;
            Nt = obj.Nt;
            Nb = obj.Nb;
            % Pass obj to generate_channels since it needs access to obj.lambda
            [H_bt, H_br, H_rt] = generate_channels(obj, Nt, Nr, Nb);
            obj.H_bt = H_bt;
            obj.H_br = H_br;
            obj.H_rt = H_rt;
            % Initialize RIS phase shifts
            rho_r = 1;
            theta = 2*pi*rand(obj.Nr,1);
            u = rho_r * exp(1j*theta);
            obj.phi = diag(u); 
            % obj.phi = eye(obj.Nr);
        end
        
        function [H_bt, H_br, H_rt] = generate_channels(obj, Nt, Nr, Nb)
            % Constants
            lambda = obj.lambda; % wavelength
            d = lambda/2; % antenna spacing
            dr = lambda/2; % element spacing for 2D arrays
            
            % Get geometric parameters
            [~,~,~,~,~,~,~,angles] = obj.computeGeometricParameters();
            
            % Generate BS-Target channel
            H_bt = generate_H_bt(obj, Nt, Nb, angles, lambda, d);
            
            % Generate BS-RIS and RIS-Target channels
            [H_br, H_rt] = generate_H_br_H_rt(obj, Nb, Nr, Nt, angles, lambda, d, dr);
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

        function a_vec = compute_a_psi(obj, Nant, psi, lambda, d)
            k = 2*pi/lambda;
            n = 0:(Nant-1);
            phase_terms = exp(1j * k * d * n * sin(psi));
            a_vec = phase_terms(:) / sqrt(Nant);
        end
        
        function a_phi = compute_a_phi(obj, Nx, phi_a, phi_e, lambda, dr)
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

        function H_Los = generate_H_Los(obj, H_bt, Nt, Nr, Nb)
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
            rho_nl = 4;
            
            % Calculate gamma_nl
            gamma_nl = sqrt(Nb*Nr)/sqrt(rho_nl);
            
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
        
        % function [next_state, reward, done] = step(obj, action)
        %     % Parse action vector - only for RIS phase shifts
        %     % Assuming action is a vector of phase shifts for each RIS element
        %     ris_phases = action(1:obj.Nr); 
            
        %     % Update RIS phase shifts (normalized between 0 and 2π)
        %     obj.phi = diag(exp(1j * 2 * pi * ris_phases));
            
        %     % Calculate performance metrics considering fixed positions
        %     % precoder = eye(obj.Nb) / sqrt(obj.Nb);  % Simple precoder
        %     [Wx,W] = computeWx(obj);
        %     covar_matrix = W * W';
        %     [peb] = obj.calculatePerformanceMetrics(covar_matrix);
        %     % peb_min = 1;
        %     % peb_max = 
        %     % Calculate reward based on communication performance
        %     R_min = computeR_min(obj);
        %     % disp(['R_min: ' num2str(R_min)]);
        %     reward = obj.computeReward(peb, obj.rate, R_min);

        %     % if reward<=0
        %     %     reward = rand(0.0,1.8);
        %     % end
            
        %     % Get next state
        %     next_state = obj.getState();
        %     % disp(['Next state size: ' num2str(size(next_state))]);
        %     % disp("no, it is not");
            
        %     % Check if episode is done (based on maximum steps or achieved performance)
        %     % You might want to modify this condition based on your requirements
        %     done = obj.isEpisodeDone();
        % end

        function [next_state, reward, done] = step(obj,action)
            % Check if episode has already terminated (e.g., out of bounds or reached destination)
            ris_phases = action(1:obj.Nr); 
            obj.phi = diag(exp(1j * 2 * pi * ris_phases));
            if isEpisodeDone(obj) || obj.stepCount >= obj.maxSteps
                done = true;
                next_state = getState(obj);
                reward = computeReward(obj);
                % info = struct();
                return;
            end
        
            % Update the vehicle's position for one time step:
            direction = (obj.destination - obj.car_loc) / norm(obj.destination - obj.car_loc);
            obj.car_loc = obj.car_loc + direction * obj.speed * obj.dt;
            obj.time = obj.time + obj.dt;
            
            % Update the target location (if the car is the dynamic target)
            obj.target_loc = obj.car_loc;
            
            % Recompute geometric parameters based on the new positions
            [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj);
            
            % Increase the step counter (helps enforce the max step limit per episode)
            obj.stepCount = obj.stepCount + 1;
            
            % Get the new state (this could be the car's position plus other sensor/angle information)
            next_state = getState(obj);  % Define getState(obj) to return your observation vector
            
            % Compute reward (for example, negative distance to destination or other criteria)
            [~,W] = computeWx(obj);
            covar_matrix = W * W';
            [peb] = obj.calculatePerformanceMetrics(covar_matrix);
            % peb_min = 1;
            % peb_max = 
            % Calculate reward based on communication performance
            R_min = computeR_min(obj);
            % disp(['R_min: ' num2str(R_min)]);
            reward = obj.computeReward(peb, obj.rate, R_min);  % Define computeReward(obj) according to your task
            
            % Check if the episode should be terminated (destination reached, out-of-bounds, or max steps reached)
            done = isEpisodeDone(obj) || (obj.stepCount >= obj.maxSteps);
            
            % Package additional info (like delays and angles)
            % info = struct('delays', delays, 'angles', angles);
        end
        

        function reward = computeReward(obj, peb, rate, R_min)
            % Compute reward based on (1/PEB) with constraint penalty
            % Parameters
            Q = 0.5; % Reward factor for unsatisfied constraints (ρ in your notation)
                        % You can adjust this value between 0 and 1
            
            % Check if constraints are satisfied
            constraints_satisfied = (rate >= R_min);
            
            % Calculate base reward as 1/PEB
            % disp(size(peb));
            base_reward = 1/peb;
            
            if ~constraints_satisfied
                % Apply penalty factor Q when constraints are not satisfied
                reward = base_reward * Q;
            else
                % Full reward when constraints are satisfied
                reward = base_reward;
            end

            % if reward <= 0
            %     reward = rand(0.0,1.8);
            % end 
        end

        function done = isEpisodeDone(obj)
            % Check if vehicle has reached the destination
            epsilon = 1.0; % Acceptable error margin in meters
            reached_dest = norm(obj.car_loc - obj.destination) < epsilon;
            
            % Check if vehicle is out of bounds
            pos = obj.car_loc;
            out_of_bounds = any(pos(1:2) < 0) || ...
                            pos(1) > obj.env_dims(1) || ...
                            pos(2) > obj.env_dims(2);
            
            % Episode ends if the vehicle reaches its destination or goes out of bounds
            done = reached_dest || out_of_bounds;
        end
        
        
        function state = reset(obj)
            % Reset simulation state
            obj.initializeChannels();
            
            % Reset vehicle state
            obj.vehicle.position = [500, 500, 0];
            obj.vehicle.velocity = [10, 0, 0];
            obj.vehicle.acceleration = [0, 0, 0];
            obj.vehicle.heading = 0;
            
            % Return the current state
            state = obj.getState();
            
        end
        % ! -------------------- MACHINE LEARNING PART ENDS HERE --------------------

        % ! -------------------- PEB COMPUTATION PART STARTS HERE --------------------
        
        % function updateVehicleDynamics(obj)
        %     % Constrain inputs
        %     while norm(obj.car_loc - obj.destination) > obj.speed * obj.dt
        %         % Move towards destination
        %         direction = (obj.destination - obj.car_loc) / norm(obj.destination - obj.car_loc);
        %         obj.car_loc = obj.car_loc + direction * obj.speed * obj.dt;
        %         obj.time = obj.time + obj.dt;
        
        %         % Update target location dynamically
        %         obj.target_loc = obj.car_loc;  % Treat car as a dynamic target
        
        %         % Compute new geometric parameters
        %         [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters(obj);
        
        %         % Pause for visualization (optional)
        %         pause(0.1);
        %     end
        % end
        

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
            
            % ? there is a delay calculation code here
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

            % BS to Target angles
            angles.bs_to_target_transmit = acos(zb/L3);
            angles.bs_to_target_receive = asin(zb/L3);
        end

        function [peb] = calculatePerformanceMetrics(obj, precoder)
            % Compute geometric parameters
            [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = obj.computeGeometricParameters();

            % Option 2: Use SVD to get dominant modes
            [U, S, V] = svd(obj.H_bt + obj.H_rt * obj.phi * obj.H_br);
            H_eff = U(:, 1:obj.Nt) * S(1:obj.Nt, 1:obj.Nt) * V(:, 1:obj.Nt)';
            
            % Check if rate constraint is satisfied
            R_min = computeR_min(obj); % Minimum required rate
            rate_constraint_satisfied = (obj.rate >= R_min);

            % Compute FIM and CRLB
            [J, ~, ~] = computeFisherInformationMatrix(obj, precoder, H_eff);
             
            % Minimize PEB by optimizing the inverse of FIM (CRLB)
            % First, ensure J is well-conditioned
            % epsilon = 1e-10;  % Small constant for numerical stability
            % J = J + epsilon * eye(size(J));


            % % Compute eigenvalue decomposition of J
            % [V, D] = eig(J);
            % eigenvalues = diag(D);

            % % Improve conditioning by adjusting smallest eigenvalues
            % min_eigenvalue = max(eigenvalues) * 1e-12;
            % eigenvalues(eigenvalues < min_eigenvalue) = min_eigenvalue;

            % % Reconstruct improved J
            % J_improved = V * diag(eigenvalues) * V';

            % Compute optimized CRLB
            CRLB = inv(J);

            % Extract position-related components (assuming first 2x2 block is position)
            % pos_CRLB = CRLB(1:2, 1:2);

            % % Optional: Apply weighting to prioritize certain dimensions
            % weights = [1, 1];  % Equal weights for x and y
            % weighted_pos_CRLB = diag(weights) * pos_CRLB * diag(weights);

            % Calculate final PEB
            peb = sqrt(trace(CRLB));
            % Optionally scale PEB based on rate constraint satisfaction
            if ~rate_constraint_satisfied
                disp("are we reaching here?")
                peb = peb * (1 + (R_min - obj.rate)/R_min);  % Penalty for rate constraint violation
            end

            % Scale PEB to be between 0-12 meters
            % initial_max_peb = 150000; % Set based on your observed values
            % peb_scaled = 12 * (peb / initial_max_peb);
            % peb_scaled = max(0, min(12, peb_scaled));

            % peb = peb_scaled;

            % Store additional metrics
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
                'SNR', obj.SNR, ...
                'Gamma', obj.gamma_c, ...
                'RateConstraintSatisfied', rate_constraint_satisfied ...
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
            
            % ? there is a delay calculation code here as well
            % TODO: VERIFY WHICH DELAY CALCULATION CODE TO USE IN FINAL
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
        
        function [dx, dy, abr] = computeAngleDerivatives(obj, angleType)
            % Numerical differentiation of angles
            epsilon = 1e-8; % Small perturbation
            % Original location
            orig_loc = obj.target_loc;
            
            % Compute derivatives using central difference method
            dx_perturb_pos = orig_loc;
            dx_perturb_pos(1) = orig_loc(1) + epsilon;
            dy_perturb_pos = orig_loc;
            dy_perturb_pos(2) = orig_loc(2) + epsilon;
            [~,~,~,~,~,~,~,angles] = obj.computeGeometricParameters();
            % Compute angles based on different types
            switch angleType
                case 'ris_to_target_aoa'
                    % Angle of Arrival at RIS-target link
                    dx = (obj.computeAngleDifference(obj.ris_loc, dx_perturb_pos, 'aoa') - ...
                        obj.computeAngleDifference(obj.ris_loc, orig_loc, 'aoa')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.ris_loc, dy_perturb_pos, 'aoa') - ...
                        obj.computeAngleDifference(obj.ris_loc, orig_loc, 'aoa')) / epsilon;
                    
                case 'bs_to_ris_response'
                    % Calculate transmitter antenna response vector (abr)
                    psi_br = angles.bs_to_ris.elevation_angle;
                    abr = obj.computeTransmitterResponseVector(psi_br);
                    
                    dx = (obj.computeAngleDifference(obj.bs_loc, dx_perturb_pos, 'transmit') - ...
                        obj.computeAngleDifference(obj.bs_loc, orig_loc, 'transmit')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.bs_loc, dy_perturb_pos, 'transmit') - ...
                        obj.computeAngleDifference(obj.bs_loc, orig_loc, 'transmit')) / epsilon;
                    
                case 'ris_receiver_response'
                    % Calculate receiver antenna response vector
                    phi_a = angles.bs_to_ris.elevation_azimuth;
                    phi_e = angle.bs_to_ris.azimuth;
                    abr = obj.compute_a_phi(phi_a, phi_e);
                    
                    dx = (obj.computeAngleDifference(obj.ris_loc, dx_perturb_pos, 'receive') - ...
                        obj.computeAngleDifference(obj.ris_loc, orig_loc, 'receive')) / epsilon;
                    dy = (obj.computeAngleDifference(obj.ris_loc, dy_perturb_pos, 'receive') - ...
                        obj.computeAngleDifference(obj.ris_loc, orig_loc, 'receive')) / epsilon;
                
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

        function abr = computeTransmitterResponseVector(obj, psi_br)
            % Calculate transmitter antenna response vector
            % psi_br: transmission angle
            N = obj.Nb; % Number of antenna elements (adjust as needed)
            d = 0.5; % Normalized antenna spacing
            k = 2 * pi; % Wave number
            
            array_positions = (0:N-1)' * d;
            abr = exp(1j * k * array_positions * sin(psi_br));
            abr = abr / sqrt(N); % Normalization
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
        
        function [Wx, W] = computeWx(obj)
            Nb = obj.Nb;  % Number of base stations
            Mb = obj.Mb;  % Number of beams
            N = obj.Ns;    % Number of subcarriers
            
            % Generate beamforming matrix W [Nb x Mb]
            W = rand(Nb, Mb) + 1j*randn(Nb, Mb);
            W = W ./ vecnorm(W);  % Normalize each column
            
            % Generate transmit data x[n] for each subcarrier n
            % x[n] is Mb x 1 complex Gaussian with zero mean and unit variance
            X = (randn(Mb, N) + 1j*randn(Mb, N)) / sqrt(2);  % Division by sqrt(2) ensures unit variance
            
            % Calculate Wx[n] for all subcarriers
            Wx = W * X;  % Results in [Nb x N] matrix where each column is Wx[n] for nth subcarrier
        end
        
        function [J, Jzao, T] = computeFisherInformationMatrix(obj, precoder, H_eff)

            [Wx,W] = computeWx(obj);
            Pb =  norm(Wx)^2; 
            B = obj.B;  % Bandwidth
            N = obj.Nt;
            SNR = log10(obj.gamma_c);
            
            sigma_s = sqrt(SNR/Pb);  % Noise variance (placeholder)
            lambda = obj.lambda;  % Wavelength
            
            % Compute geometric parameters
            [~, ~, ~, ~, ~, ~, delays, angles] = computeGeometricParameters(obj);
            
            % Estimated parameter vector
            % zeta = [
            %     delays.line_of_sight; 
            %     delays.non_line_of_sight; 
            %     angles.ris_to_target.aoa; 
            %     angles.ris_to_target.azimuth; 
            %     angles.ris_to_target.elevation_angle; 
            %     computeBSTargetAngles(obj)
            % ];
            [T, dParams] = computeTransformationMatrix(obj);
            Nb = obj.Nb;
            Nt = obj.Nt;
            [Wx,~] = computeWx(obj);
            gamma_l = sqrt(Nb*Nt)/sqrt(obj.rho_l);
            gamma_nl = sqrt(Nb*Nt)/sqrt(obj.rho_nl);
            
            [A1, A2, A3, A4] = computeAmplitudeMatrices(obj, N, B, gamma_l, gamma_nl, obj.h_l, obj.h_nl);
            [Jzao] = calculateJacobianMatrix(obj, Pb, sigma_s, N, Wx, A1, A2, A3, A4);
            
            % Compute final Fisher Information Matrix
            J = T * Jzao * T';
        end
        
        function [A1, A2, A3, A4] = computeAmplitudeMatrices(obj, N, B, gamma_l, gamma_nl, h_l, h_nl)
            [~, ~, ~, ~, ~, ~, delays, ~] = computeGeometricParameters(obj);
            tau_nl = delays.non_line_of_sight;
            tau_l = delays.line_of_sight;

            % Calculate A3 and A4 for each n
            for n = 1:N
                A3(:,n) = gamma_nl * h_nl * exp(1j * 2 * pi * B * (n/N) * delays.non_line_of_sight);
                A4(:,n) = gamma_l * h_l * exp(1j * 2 * pi * B * (n/N) * delays.line_of_sight);
            end

            % Calculate A1 and A2 for each n
            for n = 1:N
                A1(:,n) = gamma_l * h_l * 1j * 2 * pi * exp(1j * 2 * pi * B * (n/N) * delays.line_of_sight);
                A2(:,n) = gamma_nl * h_nl * 1j * 2 * pi * exp(1j * 2 * pi * B * (n/N) * delays.non_line_of_sight);
            end
        end 

        function [J_zao] = calculateJacobianMatrix(obj, Pb, sigma, N, Wx, A1, A2, A3, A4)
            % Initialize 7x7 Jacobian matrix
            J_zao = zeros(7, 7);
            
            % Calculate mu for each n
            mu = zeros(size(Wx));
            H_Los = generate_H_Los(obj, obj.H_bt, obj.Nt, obj.Nr, obj.Nb);
            % H_Los_proj = H_Los * (eye(obj.Nr)/sqrt(obj.Nr));
            HNLos = generate_H_NLoS(obj, obj.H_rt, obj.H_br, obj.Nt, obj.Nr, obj.Nb);
            % H_combined = H_Los_proj + HNLos;
            % for n = 1:N
            %     mu(:,n) = (H_Los(:,:,n) + HNLos(:,:,n)) * Wx(:,n);
            % end
            mu = (H_Los + HNLos)*Wx;
            

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
            J_zao = (2 * Pb / (sigma^2)) * J_zao;
            
        end
        
        % function [psibt, psitb] = computeBSTargetAngles(obj)
        %     % Compute BS-target transmitting and receiving angles
        %     L3 = norm(obj.bs_loc - obj.target_loc);
        %     zb = obj.bs_loc(3);
            
        %     psibt = acos(zb / L3);
        %     psitb = asin(zb / L3);
        % end

        % ! -------------------- PEB COMPUTATION PART ENDS HERE --------------------        

    end
    
end