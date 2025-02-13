% System Parameters
Nb = 8;         % Number of BS antennas (ULA)
Nt = 4;         % Number of Vehicle antennas
Nx = 8;         % Number of RIS elements in x-direction
Ny = 8;         % Number of RIS elements in y-direction
Nr = Nx * Ny;   % Total number of RIS elements
Mb = 10;        % Number of OFDM subcarriers
B = 20;         
lambda = 3e8/2e7;   % Wavelength (assuming frequency around 3GHz)
d = lambda/2;   % Inter-element spacing

% Initialize parameters
x_size = 1000;  % Environment size in meters
y_size = 1000;
z_size = 100;   % Added height dimension
dt = 0.1;       % Time step in seconds
speed = 10;     % Vehicle speed in m/s

% Initial positions
vehicle_pos = [500, 500, 0];  % Vehicle starts at ground level
goal_pos = [1000, 500, 0];    % Goal position
bs_loc = [900, 100, 20];      % Base station location
ris_loc = [200, 300, 40];     % RIS location

% Initialize channel metrics storage
channel_metrics = struct('time', [], 'direct_gain', [], 'cascaded_gain', []);

% Create ULA positions for BS (aligned along y-axis)
bs_array_positions = zeros(Nb, 3);
for i = 1:Nb
    bs_array_positions(i,:) = bs_loc + [0, (i-1)*d, 0];
end

% Create ULA positions for Vehicle (aligned along y-axis)
vehicle_array_positions = zeros(Nt, 3);
for i = 1:Nt
    vehicle_array_positions(i,:) = vehicle_pos + [0, (i-1)*d, 0];
end

% Create UPA positions for RIS (in x-y plane)
ris_array_positions = zeros(Nr, 3);
idx = 1;
for ix = 1:Nx
    for iy = 1:Ny
        ris_array_positions(idx,:) = ris_loc + [(ix-1)*d, (iy-1)*d, 0];
        idx = idx + 1;
    end
end

% Create figure
figure('Name', '3D Vehicle Simulator with Antenna Arrays');
hold on;
grid on;
axis([0 x_size 0 y_size 0 z_size]);
xlabel('X Position (m)');
ylabel('Y Position (m)');
zlabel('Z Position (m)');
title('3D Vehicle Motion Simulation with Antenna Arrays');

% Plot BS ULA
plot3(bs_array_positions(:,1), bs_array_positions(:,2), bs_array_positions(:,3), 'ks', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
plot3(bs_loc(1), bs_loc(2), bs_loc(3), 'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'r'); % BS center

% Plot RIS UPA
ris_plot = plot3(ris_array_positions(:,1), ris_array_positions(:,2), ris_array_positions(:,3), 'md', 'MarkerSize', 6, 'MarkerFaceColor', 'm');
plot3(ris_loc(1), ris_loc(2), ris_loc(3), 'md', 'MarkerSize', 12, 'MarkerFaceColor', 'r'); % RIS center

% Plot start and goal positions
plot3(vehicle_pos(1), vehicle_pos(2), vehicle_pos(3), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
plot3(goal_pos(1), goal_pos(2), goal_pos(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% Initialize vehicle array plot
vehicle_array_markers = plot3(vehicle_array_positions(:,1), vehicle_array_positions(:,2), vehicle_array_positions(:,3), 'bs', 'MarkerSize', 6, 'MarkerFaceColor', 'b');

% Draw lines connecting centers
vehicle_to_bs = plot3([vehicle_pos(1), bs_loc(1)], [vehicle_pos(2), bs_loc(2)], [vehicle_pos(3), bs_loc(3)], '--k');
vehicle_to_ris = plot3([vehicle_pos(1), ris_loc(1)], [vehicle_pos(2), ris_loc(2)], [vehicle_pos(3), ris_loc(3)], '--m');

legend('BS Array', 'BS Center', 'RIS Array', 'RIS Center', 'Start', 'Goal', 'Vehicle Array');

% Set view angle for better 3D visualization
view(45, 30);


function [received_signal, direct_channel, cascaded_channel] = simulate_downlink_transmission(bs_loc, ris_loc, vehicle_pos, W)
    
    % System Parameters
    Nb = 8;         % Number of BS antennas (ULA)
    Nt = 4;         % Number of Vehicle antennas
    Nx = 8;         % Number of RIS elements in x-direction
    Ny = 8;         % Number of RIS elements in y-direction
    Nr = Nx * Ny;   % Total number of RIS elements
    Mb = 64;  
    % Generate transmit data for each subcarrier
    x = zeros(Mb, 1);
    for n = 1:Mb
        % Generate complex Gaussian data with zero mean and unit variance
        x(n) = (randn + 1i*randn)/sqrt(2);
    end
    
    % Generate beamforming matrix if not provided
    if nargin < 4
        W = zeros(Nb, Mb);
        for i = 1:Mb
            w = (randn(Nb,1) + 1i*randn(Nb,1))/sqrt(2*Nb);
            W(:,i) = w/norm(w); % Normalize each beamforming vector
        end
    end
    
    % Verify transmit power constraint
    transmit_power = norm(W*x)^2;
    
    % Generate channels
    % Direct channel (BS to Vehicle): Nt x Nb
    H_bt = generate_H_bt(Nt, Nr);
    
    % Cascaded channel components
    % H_br (RIS to BS): Nr x Nb
    % H_rv (Vehicle to RIS): Nt x Nr
    [H_br, H_rt] = generate_H_br_H_rt(bs_loc, ris_loc, vehicle_pos, Nb, Nr, Nt);
    
    % RIS phase shifts (diagonal matrix Nr x Nr)
    Phi = diag(exp(1i*2*pi*rand(Nr,1)));
    
    % Complete cascaded channel
    cascaded_channel = H_rt * Phi * H_br; % Results in Nt x Nb
    
    % Compute received signal for each subcarrier
    received_signal = zeros(Nt, Mb);
    for n = 1:Mb
        % Direct path (Nt x 1)
        y_direct = H_bt * (W(:,n) * x(n));
        
        % RIS-assisted path (Nt x 1)
        y_ris = cascaded_channel * (W(:,n) * x(n));
        
        % Combine paths (Nt x 1)
        received_signal(:,n) = y_direct + y_ris;
    end
end

% Function to generate direct channel (Nt x Nb)
function H_bt = generate_H_bt(Nt, Nr)
    d = lambda / 2;
    [~,~,~,~,~,~,~,~, angles] = computeGeometricParameters();
    psi_bt = angles.bs_to_target.aoa_in;
    psi_tb = angles.bs_to_target.aoa_out;
    a_psi_bt = generate_a_psi_bt(Nt, psi_bt, lambda, d);
    a_psi_tb = generate_a_psi_tb(Nt, psi_bt, lambda, d);
    H_bt = a_psi_tb*a_psi_bt';
end

% Function to generate cascaded channel components
function [H_br, H_rt] = generate_H_br_H_rt(Nb, Nr, Nt)

    [~,~,~,~,~,~,~,~, angles] = computeGeometricParameters();
    psi_br = angles.bs_to_ris.azimuth;
    phi_abr = angles.bs_to_ris.elevation_azimuth;
    phi_ebr = angles.bs_to_ris.elevation_angle;

    H_br = generate_channel_with_steering(Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, lambda/2, d);

    psi_rt = angles.ris_to_target.azimuth;
    phi_art = angles.ris_to_target.elevation_azimuth;
    phi_ert = angles.ris_to_target.elevation_angle;

    H_rt = generate_channel_with_steering(Nt, Nr, phi_art, phi_ert, psi_rt, lambda, lambda/2, d);
end

function [L1, L2, L3, L_proj1, L_proj2, L_proj3, delays, angles] = computeGeometricParameters()
    
    target_loc = [500, 500, 0];  % Vehicle starts at ground level
    bs_loc = [900, 100, 20];      % Base station location
    ris_loc = [200, 300, 40]; 
    % Extract coordinates
    xb = bs_loc(1);    yb = bs_loc(2);    zb = bs_loc(3);
    xr = ris_loc(1);   yr = ris_loc(2);   zr = ris_loc(3);
    xt = target_loc(1); yt = target_loc(2); zt = target_loc(3);
    
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

    % BS to Target angles
    angles.bs_to_target.aoa_in = acos(zb/L3);
    angles.bs_to_target.aoa_out = asin(zb/L3);
end

% Function to generate uniform linear array (ULA) steering vector
function a_psi_br = generate_a_psi_br(Nant, psi, lambda, d)
    
    % Compute the array response vector
    k = 2*pi/lambda;  % Wave number
    n = 0:(Nant-1);
    
    % Steering vector computation
    phase_terms = exp(1j * k * d * n * sin(psi));
    
    % Normalize by sqrt(Nant)
    a_psi_br = phase_terms / sqrt(Nant);
    
    % Ensure column vector
    a_psi_br = a_psi_br(:);
end

function a_psi_rt = generate_a_psi_rt(Nant, psi, lambda, d)
    k = 2*pi/lambda;
    n= 0:(Nant-1);
    phase_terms = exp(1j*k*d*n*sin(psi));
    a_psi_rt =  phase_terms / sqrt(Nant);
    a_psi_rt = a_psi_rt(:);
end

% Function to generate 2D array steering vector
function a_phi_br = generate_aphi_abr(Nx, Ny, phi_a, phi_e, lambda, dr)
    % Inputs:
    % Nx: Number of elements in x-direction
    % Ny: Number of elements in y-direction
    % phi_a: Azimuth angle
    % phi_e: Elevation angle
    % lambda: Wavelength
    % dr: Element spacing
    
    % Compute total number of elements
    N2 = Nx * Ny;
    
    % Preallocate the steering vector
    a_phi_br = zeros(N2, 1);
    
    % Wave number
    k = 2*pi/lambda;
    
    % Generate 2D array response vector
    idx = 1;
    for m = 0:(Nx-1)
        for n = 0:(Ny-1)
            % Compute phase term
            phase_term = exp(1j * k * dr * (m * sin(phi_a) * sin(phi_e) + n * cos(phi_e)));
            a_phi_br(idx) = phase_term;
            idx = idx + 1;
        end
    end
    
    % Normalize by sqrt(N2)
    a_phi_br = a_phi_br / sqrt(N2);
end

function a_phi_art = generate_a_phi_art(Nx, Ny, phi_a, phi_e, lamda, dr)
    N2 = Nx*Ny;
    a_phi_art = zeros(N2, 1);
    k= 2*pi/lambda;
    idx = 1;
    for m = 0:(Nx-1)
        for n = 0:(Ny-1)
            phase_term = exp(1j*k* dr*(m*sin(phi_a)*sin(phi_e) + n*cos(phi_e)));
            a_phi_art(idx) = phase_term;
            idx = idx + 1;
        end
    end
    a_phi_art = a_phi_art / sqrt(2);
end


% Example usage
function H_br = generate_channel_with_steering(Nr, Nb, phi_abr, phi_ebr, psi_br, lambda, dr, d)
    % Generate steering vectors
    a_psi_br = generate_a_psi_br(Nr, psi_br, lambda, d);
    a_phi_abr = generate_aphi_abr(sqrt(Nb), sqrt(Nb), phi_abr, phi_ebr, lambda, dr);
    
    % Compute H_br as outer product of steering vectors
    H_br = a_phi_abr * a_psi_br';
end

function H_rt = generate_Hrt(Nt, Nr, phi_art, phi_ert, psi_rt, lambda, dr, d)
    a_psi_rt = generate_a_psi_rt(Nt, psi_rt, lambda, d);
    a_phi_art = generate_a_phi_art(sqrt(Nt), sqrt(Nr), phi_art, phi_ert, lambda, dr);

    H_rt = a_psi_rt*a_phi_art';
end 

% Function to generate LoS component based on geometry
function H_los = generate_los_component(tx_loc, rx_loc, Nr, Nt)
    % Calculate angles
    diff_vec = rx_loc - tx_loc;
    [az, el] = cart2sph(diff_vec(1), diff_vec(2), diff_vec(3));
    
    % Generate steering vectors
    H_los = exp(1i*pi*[0:Nr-1]'*sin(az)) * exp(1i*pi*[0:Nt-1]*sin(el));
    H_los = H_los/sqrt(Nr*Nt); % Normalize
end

% Main simulation loop
time = 0;
idx = 1;
while vehicle_pos(1) < goal_pos(1)
    % Update vehicle center position
    vehicle_pos(1) = vehicle_pos(1) + speed * dt;
    time = time + dt;
    
    % Update vehicle array positions
    for i = 1:Nt
        vehicle_array_positions(i,:) = vehicle_pos + [0, (i-1)*d, 0];
    end
    
    % Calculate OFDM transmission and channel characteristics
    [received_signal, direct_channel, cascaded_channel] = simulate_downlink_transmission(bs_loc, ris_loc, vehicle_pos);
    
    % Store channel metrics
    channel_metrics.time(idx) = time;
    channel_metrics.direct_gain(idx) = 20*log10(norm(direct_channel));
    channel_metrics.cascaded_gain(idx) = 20*log10(norm(cascaded_channel));
    
    % Update vehicle array visualization
    set(vehicle_array_markers, 'XData', vehicle_array_positions(:,1), ...
                              'YData', vehicle_array_positions(:,2), ...
                              'ZData', vehicle_array_positions(:,3));
    
    % Update connection lines
    set(vehicle_to_bs, 'XData', [vehicle_pos(1), bs_loc(1)], ...
                      'YData', [vehicle_pos(2), bs_loc(2)], ...
                      'ZData', [vehicle_pos(3), bs_loc(3)]);
    set(vehicle_to_ris, 'XData', [vehicle_pos(1), ris_loc(1)], ...
                       'YData', [vehicle_pos(2), ris_loc(2)], ...
                       'ZData', [vehicle_pos(3), ris_loc(3)]);
    
    % Force MATLAB to draw the update
    drawnow;
    
    % Create real-time plot of channel gains (in a separate figure)
    if mod(idx, 10) == 0  % Update plot every 10 steps to reduce computation
        figure(2);
        plot(channel_metrics.time, channel_metrics.direct_gain, 'b-', ...
             channel_metrics.time, channel_metrics.cascaded_gain, 'r-');
        grid on;
        xlabel('Time (s)');
        ylabel('Channel Gain (dB)');
        legend('Direct Channel', 'Cascaded Channel');
        title('Channel Gains vs. Time');
    end
    
    % Add small pause to make motion visible
    pause(dt);
    
    idx = idx + 1;
end

% Final analysis plots
figure(3);
subplot(2,1,1);
plot(channel_metrics.time, channel_metrics.direct_gain, 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Gain (dB)');
title('Direct Channel Gain vs. Time');

subplot(2,1,2);
plot(channel_metrics.time, channel_metrics.cascaded_gain, 'r-', 'LineWidth', 2);
grid on;
xlabel('Time (s)');
ylabel('Gain (dB)');
title('Cascaded Channel Gain vs. Time');

[received_signal, direct_channel, cascaded_channel] = simulate_downlink_transmission(bs_loc, ris_loc, vehicle_pos);

% Calculate and display channel metrics
direct_channel_gain = 20*log10(norm(direct_channel));
cascaded_channel_gain = 20*log10(norm(cascaded_channel));

fprintf('Direct channel gain: %.2f dB\n', direct_channel_gain);
fprintf('Cascaded channel gain: %.2f dB\n', cascaded_channel_gain);

% Display matrix dimensions for verification
fprintf('\nMatrix dimensions:\n');
fprintf('Direct channel: %d x %d\n', size(direct_channel));
fprintf('Cascaded channel: %d x %d\n', size(cascaded_channel));
fprintf('Received signal: %d x %d\n', size(received_signal));