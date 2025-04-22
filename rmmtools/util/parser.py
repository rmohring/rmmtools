import argparse

from typing import Any, List, Optional

import pandas as pd

from util.general import (
    listify,
    check_not_none,
    get_val_or_alt_or_raise,
    get_now_local_and_utc,
)


class CommandLineParser(argparse.ArgumentParser):
    """Standard command line parser

    Main feature is that it simplifies adding "standard command
    line options" that are used over and over.
    See the `add_std_command_line_option()` method for options.
    This version of the parser also does some validation to make sure
    the dates make sense.

    @TODO: Add the ability to specify a config.yml file.
            The command line options should override the file.
    """

    def __init__(self, description=None, opts=None):
        super().__init__(description=description)

        self._program_name = get_val_or_alt_or_raise(
            description, "A program with no description? Tsk, tsk."
        )
        self._now = get_now_local_and_utc()
        self._today_utc = self._now.utc

        self._std_opts = []
        if opts is not None:
            self.add_std_command_line_options(opts)

        return

    def get_dateint_with_offset(self, dttm=None, offset=0):
        dttm = str(dttm)
        dttm = dttm[:4] + "-" + dttm[4:6] + "-" + dttm[6:]
        dttm = pd.to_datetime(dttm).normalize()
        dttm_offset = dttm + pd.Timedelta(offset, "D")
        return int(dttm_offset.strftime("%Y%m%d"))

    def add_std_command_line_options(
        self, opts: Optional[str | List[str]] = None
    ) -> None:
        """Convenience method to add multiple default options in
        one shot. If you need to customize the help text or defaults,
        add them one at a time using `add_std_command_line_option()`.

        Args:
            opts (List[str]]): List of string shortcodes to
                add in sequence.
        """
        for opt in listify(opts):
            self.add_std_command_line_option(opt)
        return

    def add_std_command_line_option(
        self,
        opt: Optional[str] = None,
        help_text: Optional[str] = None,
        default: Any = None,
    ) -> None:
        """Add a standard option to the command line. Must be one of
        the following codes:

          `b`                  : args.begin : a begin dateint, inclusive
          `e`                  : args.end : an end dateint, EXCLUSIVE
          `d`                  : args.dateint : a single dateint
          `i`                  : args.input : an input filename
          `o`                  : args.output : an output filename
          `a`                  : args.auth_file : an authorization file
          `input_path`         : args.input_path : string value
          `output_path`        : args.output_path : string value
          `input_file_prefix`  : args.input_file_prefix : string value
          `output_file_prefix` : args.output_file_prefix : string value
          `config`             : args.config : string value
          `debug`              : args.debug : boolean True if present
          `restart`            : args.restart : boolean True if present

        Args:
            opt (str): Code for option to add.
            help_text (Optional[str], optional): Override the
                default help text used in this method.
            default (Any, optional): Override the default value used
                in this method.
        Raises:
            ValueError: if the specified option is undefined
        """
        check_not_none(opt)

        if opt == "b":
            help_text = get_val_or_alt_or_raise(
                help_text, "Begin dateint to process, inclusive"
            )
            self.add_argument(
                "-b",
                "--begin",
                type=int,
                default=default,
                metavar="BEGIN_DATEINT",
                help=help_text,
            )
        elif opt == "e":
            help_text = get_val_or_alt_or_raise(
                help_text, "End dateint to process, EXCLUSIVE"
            )
            self.add_argument(
                "-e",
                "--end",
                type=int,
                default=default,
                metavar="END_DATEINT",
                help=help_text,
            )
        elif opt == "d":
            help_text = get_val_or_alt_or_raise(
                help_text, "Single dateint to process (default: %(default)s)"
            )
            self.add_argument(
                "-d",
                "--dateint",
                type=int,
                default=self._now.dateint_utc,
                help=help_text,
            )
        elif opt == "i":
            help_text = get_val_or_alt_or_raise(
                help_text, "Input file (default: %(default)s)"
            )
            self.add_argument(
                "-i", "--input", type=str, default=default, help=help_text
            )
        elif opt == "o":
            help_text = get_val_or_alt_or_raise(
                help_text, "Output file (default: %(default)s)"
            )
            self.add_argument(
                "-o", "--output", type=str, default=default, help=help_text
            )
        elif opt == "a":
            help_text = get_val_or_alt_or_raise(
                help_text, "Authorization file (default: %(default)s)"
            )
            self.add_argument(
                "-a", "--auth_file", type=str, default=default, help=help_text
            )
        elif opt == "input_path":
            help_text = get_val_or_alt_or_raise(
                help_text, "Path for input files (default: %(default)s)"
            )
            self.add_argument("--input_path", type=str, default=default, help=help_text)
        elif opt == "input_file_prefix":
            help_text = get_val_or_alt_or_raise(
                help_text,
                'Partial input filename (default: "%(default)s" for "%(default)s_YYYYMMDD.feather")',
            )
            self.add_argument(
                "--input_file_prefix", type=str, default=default, help=help_text
            )
        elif opt == "output_path":
            help_text = get_val_or_alt_or_raise(
                help_text, "Path for output files (default: %(default)s)"
            )
            self.add_argument(
                "--output_path", type=str, default=default, help=help_text
            )
        elif opt == "config":
            help_text = get_val_or_alt_or_raise(
                help_text, "Configuration file (default: %(default)s)"
            )
            self.add_argument("--config", type=str, default=default, help=help_text)
        elif opt == "output_file_prefix":
            help_text = get_val_or_alt_or_raise(
                help_text,
                'Partial output filename (default: "%(default)s" for "%(default)s_YYYYMMDD.feather")',
            )
            self.add_argument(
                "--output_file_prefix", type=str, default=default, help=help_text
            )
        elif opt == "debug":
            help_text = get_val_or_alt_or_raise(help_text, "Turn on debug mode")
            self.add_argument("--debug", action="store_true", help=help_text)
        elif opt == "restart":
            help_text = get_val_or_alt_or_raise(help_text, "Operate in restart mode")
            self.add_argument("--restart", action="store_true", help=help_text)
        else:
            raise ValueError(f"Unknown standard option: {opt}")

        # Track which opts have been added
        self._std_opts.append(opt)
        return

    def _validate_std_args(self):
        (args, _) = self.parse_known_args()

        # Valid situations are that b < e is specified, OR just d
        if ("b" in self._std_opts) and (args.begin is not None):
            if args.end is None:
                self.error("If using -b, must also specify -e")

        if ("e" in self._std_opts) and (args.end is not None):
            if args.begin is None:
                self.error("If using -e, must also specify -b")

        # PWTNOTE: @TODO how to handle if b/d or e/d or b/e/d are
        # all specified.  For now, the below code quietly ignores the
        # -d without warning

        # if (
        #     (("b" in self._std_opts) and (args.begin is not None))
        #     or (("e" in self._std_opts) and (args.end is not None))
        # ) and (("d" in self._std_opts) and (args.dateint is not None)):
        #     print(args)
        #     self.error("Cannot specify -d at same time as -b or -e...")

        if (("b" in self._std_opts) and (args.begin is not None)) and (
            ("e" in self._std_opts) and (args.end is not None)
        ):
            if args.begin >= args.end:
                self.error("End date must be at least one day later than begin date!")
            else:
                args.dateint = None
        elif ("d" in self._std_opts) and (args.dateint is not None):
            self.set_defaults(
                begin=self.get_dateint_with_offset(args.dateint),
                end=self.get_dateint_with_offset(args.dateint, offset=1),
            )

        return args

    def parse_args(self, **kwargs) -> argparse.Namespace:
        """Override of the parent method that does validation on some
        of the standard arguments.
        """
        tmp = self._validate_std_args()
        return super().parse_args(**kwargs)


if __name__ == "__main__":
    p = CommandLineParser("fun program")
    p.add_std_command_line_options("b")
    p.add_std_command_line_options("e")
    p.add_std_command_line_options("i")
    p.add_argument(
        "-j",
        "--junk",
        type=int,
        default=None,
        metavar="JUNK_VAL",
        help="default: %(default)s",
    )
    args = p.parse_args()
    print(args)
