"""
Dependency scanner for F90/95 modules. 

The sort function provided below sorts fortran source files in order of 
increasing dependency. That is, the sorting order makes sure that modules are 
compiled before they are USE'd. 

See:
  http://scipy.org/scipy/numpy/ticket/752

The regular expressions are modified versions from the SCons.Scanner.Fortran 
module. The copyright notice for Scons is included below.

:Author: David Huard, Pearu Peterson
:Date: May, 2008
"""

""" 
Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008 The SCons Foundation 

Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the 
"Software"), to deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, merge, publish, 
distribute, sublicense, and/or sell copies of the Software, and to 
permit persons to whom the Software is furnished to do so, subject to 
the following conditions: 

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software. 

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY 
KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE 
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION 
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 
"""

import re, copy, os
import warnings

__all__ = ['FortranFileSorter', 'sort'] 

"""
Regular expression for a module name
------------------------------------

An alphanumeric word not beginning with a numeric character.
  
   [a-z_]         : a letter or an underscore
   \w*            : any number of alphanumeric characters
"""
modulename = """[a-z_]\w*"""



"""
Regular expression for a USE statement
--------------------------------------

Matches the following statements:

USE module_name
USE :: module_name
USE, INTRINSIC :: module_name
USE, NON_INTRINSIC :: module_name
USE module_name, only :: [list of procedures]


Here is a breakdown of the regex:

   (?i)               : regex is case insensitive
   (?:                : group a collection of regex symbols without saving the match as a "group"
      ^|;             : matches either the start of the line or a semicolon - semicolon
   )                  : end the unsaved grouping
   \s*                : any amount of white space
   USE                : match the string USE, case insensitive
   (?:                : group a collection of regex symbols without saving the match as a "group"
      \s+|            : match one or more whitespace OR ....  (the next entire grouped set of regex symbols)
      (?:             : group a collection of regex symbols without saving the match as a "group"
         (?:          : establish another unsaved grouping of regex symbols
            \s*          : any amount of white space
            ,         : match a comma
            \s*       : any amount of white space
            (?:NON_)? : optionally match the prefix NON_, case insensitive
            INTRINSIC : match the string INTRINSIC, case insensitive
         )?           : optionally match the ", INTRINSIC/NON_INTRINSIC" grouped expression
         \s*          : any amount of white space
         ::           : match a double colon that must appear after the INTRINSIC/NON_INTRINSIC attribute
      )               : end the unsaved grouping
   )                  : end the unsaved grouping
   \s*                : match any amount of white space
   (modulename_regex) : match the module name that is being USE'd, see above.
"""
use_regex = \
"(?i)(?:^|;)\s*USE(?:\s+|(?:(?:\s*,\s*(?:NON_)?INTRINSIC)?\s*::))\s*(%s)"%\
modulename


"""
Regular expression for a MODULE statement
-----------------------------------------

This regex finds module definitions by matching the following: 

MODULE module_name 

but *not* the following: 
 
MODULE PROCEDURE procedure_name 
 
Here is a breakdown of the regex: 
   (?i)               : regex is case insensitive
   ^\s*               : any amount of white space 
   MODULE             : match the string MODULE, case insensitive 
   \s+                : match one or more white space characters 
   (?!PROCEDURE)      : but *don't* match if the next word matches 
                        PROCEDURE (negative lookahead assertion), 
                        case insensitive 
   ([a-z_]\w*)        : match one or more alphanumeric characters 
                        that make up the defined module name and 
                        save it in a group 
"""
def_regex = """(?i)^\s*MODULE\s+(?!PROCEDURE)(%s)"""%modulename 


"""
Regular expression for an INCLUDE statement
-------------------------------------------
The INCLUDE statement regex matches the following:

   INCLUDE 'some_Text'
   INCLUDE "some_Text"
   INCLUDE "some_Text" ; INCLUDE "some_Text"
   INCLUDE kind_"some_Text"
   INCLUDE kind_'some_Text"

where some_Text can include any alphanumeric and/or special character
as defined by the Fortran 2003 standard.

Here is a breakdown of the regex:

   (?i)               : regex is case insensitive
   (?:                : begin a non-saving group that matches the following:
      ^               :    either the start of the line
      |               :                or
      ['">]\s*;       :    a semicolon that follows a single quote,
                           double quote or greater than symbol (with any
                           amount of whitespace in between).  This will
                           allow the regex to match multiple INCLUDE
                           statements per line (although it also requires
                           the positive lookahead assertion that is
                           used below).  It will even properly deal with
                           (i.e. ignore) cases in which the additional
                           INCLUDES are part of an in-line comment, ala
                                           "  INCLUDE 'someFile' ! ; INCLUDE 'someFile2' "
   )                  : end of non-saving group
   \s*                : any amount of white space
   INCLUDE            : match the string INCLUDE, case insensitive
   \s+                : match one or more white space characters
   (?\w+_)?           : match the optional "kind-param _" prefix allowed by the standard
   [<"']              : match the include delimiter - an apostrophe, double quote, or less than symbol
   (.+?)              : match one or more characters that make up
                        the included path and file name and save it
                        in a group.  The Fortran standard allows for
                        any non-control character to be used.  The dot
                        operator will pick up any character, including
                        control codes, but I can't conceive of anyone
                        putting control codes in their file names.
                        The question mark indicates it is non-greedy so
                        that regex will match only up to the next quote,
                        double quote, or greater than symbol
  (?=["'>])          : positive lookahead assertion to match the include
                        delimiter - an apostrophe, double quote, or
                       greater than symbol.  This level of complexity
                        is required so that the include delimiter is
                        not consumed by the match, thus allowing the
                        sub-regex discussed above to uniquely match a
                        set of semicolon-separated INCLUDE statements
                        (as allowed by the F2003 standard)
"""
include_regex = \
    """(?i)(?:^|['">]\s*;)\s*INCLUDE\s+(?:\w+_)?[<"'](.+?)(?=["'>])"""


"""
Regular expression for a comment
--------------------------------

One limitation of the original Scons scanner is that it cannot properly USE 
statements if they are commented out. In either of the following cases:

   !  USE mod_a ; USE mod_b         [entire line is commented out]
   USE mod_a ! ; USE mod_b       [in-line comment of second USE statement]

the second module name (mod_b) will be picked up as a dependency even though 
it should be ignored. The proposed solution is to first parse the file to 
remove all the comments.

(^.*)!?             : match everything on a line before an optional comment
.*$                 : ignore the rest of the line
"""
comment_regex = r'(^.*)!?.*$'
    
class FortranFileSorter:
    """Given a list of fortran 90/95 files, return a file list sorted by module
    dependency. If a file depends on a parent through a USE MODULE statement, 
    this parent file will occur earlier in the list. 
    
    Parameters
    ----------
    files : sequence
      The sequence of file names to be sorted.
    
    """
    def __init__(self, files):
        self.files = files
        self.use_regex = re.compile(use_regex, re.MULTILINE)
        self.def_regex = re.compile(def_regex, re.MULTILINE)
        self.comment_regex = re.compile(comment_regex, re.MULTILINE)
        self.include_regex = re.compile(include_regex, re.MULTILINE)

    def read_code(self, file):
        """Open the file and return the code as one string."""
        if not hasattr(file, 'readlines'):
            file = open(file, 'r')
        return file.read()

    def uncomment(self, code):
        """Return an uncommented version of a code string."""
        return '\n'.join(self.comment_regex.findall(code))

    def match_used(self, code):
        """Return the set of used module in a code string."""
        return set(map(str.lower, self.use_regex.findall(code)))

    def match_defined(self, code):
        """Return the set of defined module in a code string."""
        return set(map(str.lower, self.def_regex.findall(code)))
        
    def match_included(self, code):
        """Return the set of included files in a code string."""
        return set(self.include_regex.findall(code))
        
    def externals(self):
        """Return the modules that are used but not defined in the list of 
        files."""
        if not hasattr(self, 'mod_defined'):
            self.scan()
        all_used = reduce(set.union, self.mod_used.values())
        all_defined = reduce(set.union, self.mod_defined.values())
        return all_used.difference(all_defined)
        
    def scan(self):
        """For each file, identify the set of modules that are
         defined, used but not defined in the same file, and the set of 
         included files.
        """
        self.mod_defined = {}
        self.mod_used = {}
        self.included = {}
        for f in self.files:
            code = self.read_code(f)
            code = self.uncomment(code)         
            self.mod_defined[f] = self.match_defined(code)
            self.mod_used[f] = self.match_used(code)
            self.included[f] = self.match_included(code)
            
            # Remove from used the modules defined internally. 
            # That is, keep elements in used but not in defined. 
            self.mod_used[f].difference_update(self.mod_defined[f]) 
                    
    # TODO : Deal with include (do we have too?)                        
    def sort(self):
        """Sort the files in order of dependency. 
        """      
        ordered_list = []
        
        # Take care of modules not defined anywhere in the files. eg.
        # an intrinsic module by assuming them known from the 
        # start.
        defined = self.externals()
            
        # This cycles through the files, and appends those whose dependencies 
        # are already satisfied by the files in the ordered list. 
        remaining = set(self.files)
        goon = True
        while goon:
            goon = False
            for f in remaining:
                dependencies_satisfied = self.mod_used[f].issubset(defined)
                if dependencies_satisfied:
                    ordered_list.append(f)            
                    defined.update(self.mod_defined[f])
                    goon = True
            remaining.difference_update(set(ordered_list))

        ordered_list.extend(list(remaining))
        return ordered_list
    
    
    
def sort(files):
    """Given a list of fortran 90/95 files, return a file list sorted by module
    dependency. If a file depends on a parent through a USE MODULE statement, 
    this parent file will occur earlier in the list. 
    
    Parameters
    ----------
    files : sequence
      The sequence of file names to be sorted.
    """
    FS = FortranFileSorter(files)
    FS.scan()
    return FS.sort() 


import sys
if __name__ == "__main__":
    print sys.argv[1:]
    sort(sys.argv[1:])



