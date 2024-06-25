-- Stimulus process
stim_proc: process
begin
	rest <= '1';
	wait for 100 ns;
	
	rest <= '0';
	we <= '1';
	w_addr <= "0000000";
	w_data <= X"00000001";
	wait for 100 ns;
	
	w_addr <= "0000001";
	w_data <= X"00000004";
	wait for 100 ns;
	
	w_addr <= "0000010";
	w_data <= X"00000014";
	wait for 100 ns;
	
	w_addr <= "0000011";
	w_data <= X"0000000F";
	wait for 100 ns;
	
	we <= '0';
	r_addr <= "0000000";
	wait for 100 ns;
	
	r_addr <= "0000001";
	wait for 100 ns;
	
	r_addr <= "0000010";
	wait for 100 ns;
	
	r_addr <= "0000011";
	wait for 100 ns;
wait;
end process;
