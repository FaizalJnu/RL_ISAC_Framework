
classdef channel_functions
    properties
                % Initial positions
        vehicle_pos = [500, 500, 0];  % Vehicle starts at ground level
        goal_pos = [1000, 500, 0];    % Goal position
        bs_loc = [900, 100, 20];      % Base station location
        ris_loc = [200, 300, 40];     % RIS location

        % System Parameters
        Nb = 8;         % Number of BS antennas (ULA)
        Nt = 4;         % Number of Vehicle antennas
        Nx = 8;         % Number of RIS elements in x-direction
        Ny = 8;         % Number of RIS elements in y-direction

        fc = 28e9;
        c = physconst('lightspeed');
        lambda = c/fc;
        rng(1000,1000);

        % Setup surface
        Nr = 10;
        Nc = 20;
        dr = 0.5*lambda;
        dc = 0.5*lambda;

    end

    % TODO: We need to implement all the channel functions here and possibly
    % TODO: the helper function
    methods
        function obj = ClassName(inputArg1)
            obj.Property1 = inputArg1^2;
            disp(obj.Property1);
            
        end
        % construct surface
        ris = helperRISSurface('Size',[Nr Nc],'ElementSpacing',[dr dc],...
            'ReflectorElement',phased.IsotropicAntennaElement,'OperatingFrequency',fc);
    end
end
