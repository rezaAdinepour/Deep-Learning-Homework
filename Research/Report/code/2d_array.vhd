constant W: natural := 7; -- number of bits in address
constant B: natural := 32; -- number of bits in data

type reg_file_type is array(2 ** W-1 downto 0) of
		std_logic_vector(B-1 downto 0);

signal array_reg: reg_file_type;
signal array_next: reg_file_type;
signal en: std_logic_vector(2 ** W-1 downto 0);
