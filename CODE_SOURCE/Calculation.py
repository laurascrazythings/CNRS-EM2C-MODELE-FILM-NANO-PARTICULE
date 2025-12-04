import datetime as dtm


def runtime_program (beginning_date_and_time):

    """Calculates and prints program runtime based on start time.

    Parameters
    ----------
    beginning_date_and_time : datetime.datetime
    
    Returns
    -------
    None
    """

    ending_date_and_time=dtm.datetime.now()
    runtime = ending_date_and_time - beginning_date_and_time

    #type(runtime) =  datetime.timedelta, it should be convert in a string to be easily manipulated and printed
    string_runtime = str(runtime)

    hour = string_runtime.split(":")[0]
    minute = string_runtime.split(":")[1]
    second = str(round(float(string_runtime.split(":")[2]),2))

    # These different cases are made to avoid printing 00h00min00,4s for the shortest program (just for the visual aspect)
    if hour == "0":

        if minute == "00":

            print("Runtime : "+second+"s")

        else:

            print("Runtime : "+minute+"min"+second+"s")
    
    else:

        print("Runtime : "+hour+"h"+minute+"min"+second+"s")
    
    return None
