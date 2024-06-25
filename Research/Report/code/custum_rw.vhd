-- Stimulus process
stim_proc: process
begin
	-- Write phase
	we <= '1';
	adr <= "0000000";
	din <= X"FFFFFFFF";
	wait for 100 ns;

	adr <= "1001101";
	din <= X"0000000A";
	wait for 100 ns;

	adr <= "0000001";
	din <= X"0000FFFF";
	wait for 100 ns;

	adr <= "0000111";
	din <= X"00000A0F";
	wait for 100 ns;

	adr <= "1000001";
	din <= X"F0000000";
	wait for 100 ns;



	-- Read phase
	we <= '0';

	adr <= "0000111";
	wait for 100 ns;

	adr <= "0000111";
	wait for 100 ns;

	adr <= "0000000";
	wait for 100 ns;

	adr <= "0000001";
	wait for 100 ns;

	adr <= "1000001";
	wait for 100 ns;

	adr <= "1001101";
	wait for 100 ns;
end process;
