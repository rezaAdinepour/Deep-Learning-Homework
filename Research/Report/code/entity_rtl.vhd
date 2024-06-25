entity reg_file is
	port( clk, rest: in std_logic;
	      we: in std_logic;
	      w_addr: in std_logic_vector(6 downto 0);
	      w_data: in std_logic_vector(31 downto 0);
	      r_addr: in std_logic_vector(6 downto 0);
	      r_data: out std_logic_vector(31 downto 0) );
end reg_file;
