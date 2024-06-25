-- Stimulus process
stim_proc: process
begin
	for i in 0 to (2 ** adr'length-1) loop
		we <= '1';
		adr <= std_logic_vector(to_unsigned(i, adr'length));
		din <= std_logic_vector(to_unsigned(0, din'length));
		wait for 100 ns;
	end loop;
	wait;
end process;
