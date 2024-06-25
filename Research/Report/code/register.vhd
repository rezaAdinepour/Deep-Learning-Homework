-- register
process(clk, rest)
begin

	if(rest = '1') then
		for i in W+1 downto 0 loop
			array_reg(i) <= (others => '0');
		end loop;
--			array_reg(3) <= (others => '0');
--			array_reg(2) <= (others => '0');
--			array_reg(1) <= (others => '0');
--			array_reg(0) <= (others => '0');
		
	elsif(rising_edge(clk)) then
		for i in W+1 downto 0 loop
			array_reg(i) <= array_next(i);
		end loop;
--			array_reg(3) <= array_next(3);
--			array_reg(2) <= array_next(2);
--			array_reg(1) <= array_next(1);
--			array_reg(0) <= array_next(0);
	end if;
end process;
